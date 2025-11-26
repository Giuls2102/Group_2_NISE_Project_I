using System.Collections;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using System.Text.RegularExpressions;
using System.Globalization;
using TMPro;

[DisallowMultipleComponent]
[RequireComponent(typeof(AudioSource))]
public class SequenceTablesFeedback : MonoBehaviour
{
    [Header("Tables (order must match zones)")]
    public Transform[] tableRoots = new Transform[3];

    [Header("Zone names from Python (must match)")]
    public string[] tableZones = new[] { "left", "center", "right" };

    [Header("Colors")]
    public Color okColor  = new Color(0.20f, 0.80f, 0.40f);
    public Color errColor = new Color(0.85f, 0.25f, 0.25f);
    [Tooltip("Red flash / shake duration on wrong placement")]
    public float flashSeconds = 0.6f;

    [Header("Shake (on WRONG)")]
    [SerializeField] float shakePosAmp = 0.01f;  // ~1 cm
    [SerializeField] float shakeRotAmp = 2.5f;   // degrees (roll)
    [SerializeField] float shakeHz     = 22f;    // Hz

    [Header("Books")]
    public GameObject[] bookRoots;
    public float bookAppearDelay = 0.25f;
    public bool hideBooksOnComplete = false;
    
    [SerializeField] int targetsPerTrial = 3;
    [Header("Arduino")]
    public ArduinoRecorderBridge arduino;
    
    [Header("Aruco")]
	public PythonArucoBridge arucoBridge;
	
	[Header("Mediapipe")]
    public MediapipeBridge mediapipe;

    [Header("Blackboard (instructions)")]
    public TMP_Text instructionText;                         // e.g., BlackboardText
    [TextArea] public string instructionMessage =
        "Bitte hilf den Schülern, die Bücher in der angegebenen Reihenfolge zu ordnen.";
    public bool showInstructionsOnBegin = true;
    public bool hideInstructionsOnComplete = false;

    [Header("Pink screen (Attempts/Time)")]
    public TMP_Text smallBoardText;                          // e.g., Poster ▸ Scores

    [Header("‘Well done’ overlay (optional)")]
    public TMP_Text messageText;
    [SerializeField] string doneMessage = "Sehr gut!";
    [SerializeField] float messageSeconds = 1.2f;

    [Header("Audio")]
    public AudioClip highClip;   // correct ding
    public AudioClip lowClip;    // wrong buzz
    [Range(0f,1f)] public float highVol = 0.8f;
    [Range(0f,1f)] public float lowVol  = 0.8f;
	
	
	[Header("Trials")]
	public int currentTrialNumber = 0;                 // set by Menu when starting
	public TrialHistoryManager historyManager;         // drag in Inspector (or it will auto-find)
	
    [Header("After completion / return")]
    [SerializeField] bool resetTablesOnComplete   = true;
    //[SerializeField] bool hideSmallBoardOnComplete = true;
    [SerializeField] bool autoReturnToMenu        = true;
    [SerializeField] bool autoOpenHistoryInstead  = false;
    [SerializeField] float afterCompleteDelay     = 1.0f;
    [SerializeField] MenuController menuController; // drag your MenuController here

    [Header("Debug")]
    public bool debugLogs = true;

    // ----- internals -----
    AudioSource audioSource;
    List<Renderer>[] tableRends;
    readonly Dictionary<Renderer, Color> original = new Dictionary<Renderer, Color>();
    readonly Dictionary<string,int> zoneToIndex   = new Dictionary<string, int>();
    readonly HashSet<string> _bookLogged = new HashSet<string>(System.StringComparer.OrdinalIgnoreCase);

    bool[] zoneGreen;                 // per table/zone
    Coroutine[] shakeCo;              // per table shake guard
    Coroutine bookAppearCo;

    int   totalWrong = 0;             // wrong events seen
    float totalStart = -1f;           // first feedback timestamp
    bool  finished   = false;

    void Awake()
    {
        // Audio
        audioSource = GetComponent<AudioSource>();
        audioSource.playOnAwake = false;
        audioSource.spatialBlend = 0f;

        // Map zones
        zoneToIndex.Clear();
        for (int i = 0; i < tableZones.Length; i++)
        {
            var key = Normalize(tableZones[i]);
            if (!string.IsNullOrEmpty(key) && !zoneToIndex.ContainsKey(key))
                zoneToIndex[key] = i;
        }

        // Collect renderers
        tableRends = new List<Renderer>[tableRoots.Length];
        for (int i = 0; i < tableRoots.Length; i++)
        {
            tableRends[i] = new List<Renderer>();
            if (tableRoots[i]) tableRends[i].AddRange(tableRoots[i].GetComponentsInChildren<Renderer>(true));
            else LogWarn($"tableRoots[{i}] not assigned.");
        }
        CacheOriginals(tableRends);

        // Fallback beeps
        if (!highClip) highClip = MakeBeep(880f, 0.12f, highVol);
        if (!lowClip)  lowClip  = MakeBeep(330f, 0.15f, lowVol);

        // Auto-bind texts if not assigned
        if (!instructionText) AutoBindInstruction();   // blackboard
        if (!smallBoardText)  AutoBindSmallBoard();    // pink screen

        if (messageText) messageText.gameObject.SetActive(false);
        SetBooksActive(false);

        zoneGreen = new bool[tableRoots.Length];
        shakeCo   = new Coroutine[tableRoots.Length];
        Log("[Sequence] Awake OK");
    }
	

	float ExtractFloat(string s, string key)
	{
		var m = Regex.Match(s, key + @"[^0-9\-]*(-?\d+(?:\.\d+)?)");
		if (!m.Success) return -1f;
		if (float.TryParse(m.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
			return v;
		return -1f;
	}
    
    public void BeginTrial()
    {
        Log("[Sequence] BeginTrial()");
        enabled = true;              // ensure this script is active after returning from menu
        finished   = false;
        totalWrong = 0;
        totalStart = -1f;
        _bookLogged.Clear();

        // stop shakes & book co
        for (int i = 0; i < shakeCo.Length; i++) { if (shakeCo[i] != null) StopCoroutine(shakeCo[i]); shakeCo[i] = null; }
        if (bookAppearCo != null) { StopCoroutine(bookAppearCo); bookAppearCo = null; }

        // reset visuals
        ResetTablesToOriginal();

        // Blackboard: show instructions
        if (showInstructionsOnBegin && (instructionText || AutoBindInstruction()))
        {
            if (!string.IsNullOrEmpty(instructionMessage))
                instructionText.text = instructionMessage;
            instructionText.gameObject.SetActive(true);
        }
        

        // Pink screen: live attempts + time placeholder
        if (smallBoardText || AutoBindSmallBoard())
        {
            smallBoardText.text = ComposeSmallBoardLive(); // Attempts: N / Time: --
            smallBoardText.gameObject.SetActive(true);
        }
				
		if (!historyManager) historyManager = TrialHistoryManager.Instance;
		var hm = TrialHistoryManager.Instance;
		int trialNo = hm ? hm.NextTrialNumber() : 1;
		

		// Assign trial number if not provided
		if (currentTrialNumber <= 0)
			currentTrialNumber = historyManager ? historyManager.NextTrialNumber() : 1;
		if (arduino)
		{
			// Ensure the background python is alive, then start a new trial recording
			arduino.StartBridge();                   // safe if already running
			arduino.StartRecording(currentTrialNumber);
		}
		
		if(mediapipe != null)
		{
			mediapipe.StartRecording(currentTrialNumber);
		}
		// Pink screen initial text
		if (smallBoardText) {
			smallBoardText.text = $"Durchgang {currentTrialNumber}\nVersuche: 0\nZeit: --";
			smallBoardText.gameObject.SetActive(true);
		}
        // Books appear shortly after Start
        SetBooksActive(false);
        bookAppearCo = StartCoroutine(ShowBooksAfterDelay(bookAppearDelay));
    }

    // -------- Events from Python bridge --------

    public void OnArucoCorrect(string objectName, string zoneName)
    {
        if (finished) return;
        if (totalStart < 0f) totalStart = Time.realtimeSinceStartup;

        int idx = ZoneToIndex(Normalize(zoneName));
        if (idx >= 0)
        {
            zoneGreen[idx] = true;
            PaintGroup(tableRends[idx], okColor);
        }

        Play(highClip, highVol);
        UpdateSmallBoardLive();

        if (AllZonesGreen())
        {
            finished = true;
            StartCoroutine(ShowSummaryAndReturn());
        }
    }

    public void OnArucoWrong(string objectName, string wrongZone, string expectedZone)
    {
        if (finished) return;
        if (totalStart < 0f) totalStart = Time.realtimeSinceStartup;

        totalWrong += 1;

        int idxWrong = ZoneToIndex(Normalize(wrongZone));
        if (idxWrong >= 0)
        {
            if (shakeCo[idxWrong] != null) StopCoroutine(shakeCo[idxWrong]);
            shakeCo[idxWrong] = StartCoroutine(FlashWrongAndShake(idxWrong));
        }

        Play(lowClip, lowVol);
        UpdateSmallBoardLive();
    }

    // -------- Completion flow --------

    IEnumerator ShowSummaryAndReturn()
	{
		int   finalAttempts = totalWrong + targetsPerTrial;
		float totalTime     = (totalStart >= 0f) ? (Time.realtimeSinceStartup - totalStart) : 0f;

		// Update pink board (live summary -> final)
		if (smallBoardText)
		{
			smallBoardText.text = $"Durchgang {currentTrialNumber}\nVersuche: {finalAttempts}\nZeit: {totalTime:0.00} s";
			smallBoardText.gameObject.SetActive(true);
		}
		

		// Optional 'Well done'
		if (messageText)
		{
			messageText.text = doneMessage;
			messageText.gameObject.SetActive(true);
		}
		yield return new WaitForSecondsRealtime(messageSeconds);

		if (messageText) messageText.gameObject.SetActive(false);

		if (resetTablesOnComplete) ResetTablesToOriginal();
		if (hideBooksOnComplete)   SetBooksActive(false);
		HideAllUI(); // hides instructions & pink text
		if (arduino) arduino.StopRecording();
		if (mediapipe) mediapipe.OnTrialEnd();
		if (autoReturnToMenu && EnsureMenu())
		{
			yield return new WaitForSecondsRealtime(afterCompleteDelay);
			if (autoOpenHistoryInstead) menuController.ShowHistory();
			else                        menuController.ToMenu();
			enabled = false;
		}
	}


    // -------- Live pink-screen updates --------

    void UpdateSmallBoardLive()
    {
        if (!smallBoardText && !AutoBindSmallBoard()) return;
        smallBoardText.text = ComposeSmallBoardLive();
        if (!smallBoardText.gameObject.activeSelf) smallBoardText.gameObject.SetActive(true);
    }

    string ComposeSmallBoardLive()
    {
        int attemptsNow = totalWrong + SolvedCount();
        return $"Attempts: {attemptsNow}\nTime: --";
    }

    int SolvedCount()
    {
        int c = 0;
        for (int i = 0; i < zoneGreen.Length; i++) if (zoneGreen[i]) c++;
        return c;
    }

    // -------- Helpers -------

    int ZoneToIndex(string z)
    {
        if (string.IsNullOrEmpty(z)) return -1;
        return zoneToIndex.TryGetValue(z.Trim().ToLowerInvariant(), out var idx) ? idx : -1;
    }

    bool AllZonesGreen()
    {
        for (int i = 0; i < zoneGreen.Length; i++) if (!zoneGreen[i]) return false;
        return true;
    }

    IEnumerator FlashWrongAndShake(int i)
    {
        if (i < 0 || i >= tableRends.Length || tableRoots[i] == null || tableRends[i].Count == 0) yield break;

        PaintGroup(tableRends[i], errColor);

        var t  = tableRoots[i];
        Vector3 basePos = t.localPosition;
        Quaternion baseRot = t.localRotation;

        float t0 = Time.realtimeSinceStartup;
        float twoPiF = 2f * Mathf.PI * shakeHz;

        while (Time.realtimeSinceStartup - t0 < flashSeconds)
        {
            float tt = Time.realtimeSinceStartup - t0;
            float sx = Mathf.Sin(twoPiF * tt);
            float sz = Mathf.Sin(twoPiF * tt + 1.2f);

            t.localPosition = basePos + new Vector3(sx * shakePosAmp, 0f, sz * shakePosAmp);
            t.localRotation = baseRot * Quaternion.Euler(0f, 0f, sx * shakeRotAmp);
            yield return null;
        }

        t.localPosition = basePos;
        t.localRotation = baseRot;

        if (zoneGreen[i]) PaintGroup(tableRends[i], okColor);
        else RestoreGroup(tableRends[i]);

        shakeCo[i] = null;
    }

    IEnumerator ShowBooksAfterDelay(float d)
    {
        float t0 = Time.realtimeSinceStartup;
        while (Time.realtimeSinceStartup - t0 < d) yield return null;
        SetBooksActive(true);
        bookAppearCo = null;
    }

    void SetBooksActive(bool on)
    {
        foreach (var b in bookRoots) if (b) b.SetActive(on);
    }

    // --- colors ---
    void CacheOriginals(List<Renderer>[] groups)
    {
        foreach (var list in groups)
            foreach (var r in list)
                if (r && !original.ContainsKey(r))
                    original[r] = GetBaseColor(r);
    }

    void PaintGroup(List<Renderer> list, Color c)
    {
        foreach (var r in list) if (r) SafeSetColor(r, c);
    }

    void RestoreGroup(List<Renderer> list)
    {
        foreach (var r in list)
            if (r && original.TryGetValue(r, out var col))
                SafeSetColor(r, col);
    }

    static Color GetBaseColor(Renderer r)
    {
        var m = r.sharedMaterial; if (!m) return Color.white;
        if (m.HasProperty("_BaseColor")) return m.GetColor("_BaseColor"); // URP Lit
        if (m.HasProperty("_Color"))     return m.GetColor("_Color");     // Standard/Built-in
        return Color.white;
    }

    static void SafeSetColor(Renderer r, Color c)
    {
        var m = r.material; if (!m) return; // instance
        if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", c);
        else if (m.HasProperty("_Color")) m.color = c;
    }

    // --- audio ---
    void Play(AudioClip clip, float vol = 1f)
    {
        if (!clip || !audioSource) return;
        audioSource.spatialBlend = 0f;
        audioSource.volume = Mathf.Clamp01(vol);
        audioSource.PlayOneShot(clip);
    }

    AudioClip MakeBeep(float f, float d, float v)
    {
        int sr = 44100, n = Mathf.CeilToInt(sr * d);
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = v * Mathf.Sin(2 * Mathf.PI * f * i / sr);
        var clip = AudioClip.Create($"beep_{f}", n, 1, sr, false);
        clip.SetData(data, 0);
        return clip;
    }

    // --- auto-bind helpers ---
    TMP_Text AutoBindInstruction()
    {
        if (instructionText) return instructionText;
        string[] candidates = { "BlackboardText", "Instructions", "InstructionText" };
        foreach (var t in Resources.FindObjectsOfTypeAll<TMP_Text>())
        {
            foreach (var name in candidates)
            {
                if (string.Equals(t.name, name, System.StringComparison.OrdinalIgnoreCase))
                {
                    instructionText = t;
                    if (!t.gameObject.activeSelf) t.gameObject.SetActive(true);
                    Log("[Sequence] Bound instructionText to: " + GetPath(t.transform));
                    return instructionText;
                }
            }
        }
        LogWarn("[Sequence] Blackboard TMP_Text not assigned and not found.");
        return null;
    }

    TMP_Text AutoBindSmallBoard()
    {
        if (smallBoardText) return smallBoardText;
        string[] candidates = { "Scores", "PinkBoardText", "PosterText" };
        foreach (var t in Resources.FindObjectsOfTypeAll<TMP_Text>())
        {
            foreach (var name in candidates)
            {
                if (string.Equals(t.name, name, System.StringComparison.OrdinalIgnoreCase))
                {
                    smallBoardText = t;
                    if (!t.gameObject.activeSelf) t.gameObject.SetActive(true);
                    Log("[Sequence] Bound smallBoardText to: " + GetPath(t.transform));
                    return smallBoardText;
                }
            }
        }
        LogWarn("[Sequence] Pink-screen TMP_Text not assigned and not found.");
        return null;
    }

    string GetPath(Transform t)
    {
        var parts = new List<string>();
        while (t != null) { parts.Add(t.name); t = t.parent; }
        parts.Reverse();
        return string.Join("/", parts);
    }

    bool EnsureMenu()
    {
        if (!menuController) menuController = FindFirstObjectByType<MenuController>();
        return menuController != null;
    }
    
    public void ResetTablesToOriginal() {
		foreach (var kv in original) SafeSetColor(kv.Key, kv.Value);
		for (int i = 0; i < zoneGreen.Length; i++) zoneGreen[i] = false;
	}

	public void HideAllUI() {
		if (instructionText) instructionText.gameObject.SetActive(false);
		if (smallBoardText)  smallBoardText.gameObject.SetActive(false);
		SetBooksActive(false);
	}

	public void ShowPinkBoard(bool on)
	{
		if (smallBoardText) smallBoardText.gameObject.SetActive(on);
	}
    // utils
    string Normalize(string s) => string.IsNullOrEmpty(s) ? "" : s.Trim().ToLowerInvariant();
    void Log(string msg)    { if (debugLogs) Debug.Log(msg); }
    void LogWarn(string msg){ if (debugLogs) Debug.LogWarning(msg); }
}

	
	
