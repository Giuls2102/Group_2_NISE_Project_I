using System.Collections;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Playables;
using UnityEngine.UI;

public class MenuController : MonoBehaviour
{
    [Header("UI — Menu")]
    public Canvas menuCanvas;
    public Button easyButton;
    public Button mediumButton;
    public Button hardButton;
    public Button historyButton;

    [Header("UI — History")]
    public Canvas historyCanvas;               // Root History canvas
    public CanvasGroup historyCanvasGroup;     // Optional (fade / raycast)
    public Button backButton;                  // Optional (ESC works too)
    
    public enum Difficulty { Easy, Medium, Hard }

	[Header("Difficulty")]
	[SerializeField] Difficulty defaultDifficulty = Difficulty.Easy;

    [Header("Timing")]
    [Tooltip("Frames to wait before showing the first screen (e.g., door animation).")]
    public int framesToWait = 7000;
    [Tooltip("If true, open History after the wait instead of the Menu.")]
    public bool openHistoryAtStartup = false;

    [Header("Gameplay")]
    [Tooltip("Assign the object with SequenceTablesFeedback.")]
    public SequenceTablesFeedback sequence;    // single source of truth
    [Tooltip("Optional Python bridge for camera/ArUco.")]
    public PythonArucoBridge arucoBridge;      // optional
    [Tooltip("Optional Timeline that zooms to the pink board for History.")]
    public PlayableDirector historyZoomDirector; // optional
    [Tooltip("Optional Python bridge for camera/ArUco.")]
    public ArduinoRecorderBridge arduinoRecorder;      // optional

    [Header("History Presenter (optional)")]
    [Tooltip("Any MonoBehaviour that has a method named 'ComputeAndShow' (will be SendMessage'd).")]
    [SerializeField] HistoryList historyPresenter;     // e.g., CsvArduinoViewer

    [Header("Button visuals")]
    [SerializeField] Color easyidleColor    = new Color(0.2f, 0.85f, 0.20f, 1.0f); // yellow
    [SerializeField] Color mediumidleColor    = new Color(1.00f, 0.55f, 0.0f, 1.0f);
    [SerializeField] Color hardleColor    = new Color(1.00f, 0.0f, 0.00f, 1.0f);
    [SerializeField] Color historyleColor    = new Color(0.00f, 0.45f, 1.0f, 1.0f);
    [SerializeField] Color pressedColor = new Color(0.20f, 0.80f, 0.40f); // green
    [SerializeField] float feedbackSeconds = 0.25f;

    bool menuVisible = false;

    // ---------------- lifecycle ----------------
    void Awake()
    {
        // Start hidden; we decide after the wait
        SafeSetCanvasVisible(menuCanvas,   false);
        SafeSetCanvasVisible(historyCanvas, false);

        if (historyCanvasGroup)
        {
            historyCanvasGroup.alpha = 0f;
            historyCanvasGroup.interactable = false;
            historyCanvasGroup.blocksRaycasts = false;
        }

        if (sequence) sequence.enabled = false;

        if (easyButton)   easyButton.onClick.AddListener(() => StartGame(Difficulty.Easy));
        if (mediumButton)   mediumButton.onClick.AddListener(() => StartGame(Difficulty.Medium));
        if (hardButton)   hardButton.onClick.AddListener(() => StartGame(Difficulty.Hard));
        if (historyButton) historyButton.onClick.AddListener(ShowHistory);
        if (backButton)    backButton.onClick.AddListener(BackFromHistory);
    }
    

    IEnumerator Start()
    {
        for (int i = 0; i < Mathf.Max(0, framesToWait); i++) yield return null;
        if (openHistoryAtStartup) ShowHistory();
        else                      ToMenu();
    }

    void Update()
    {
        var kb = Keyboard.current; if (kb == null) return;

        // ESC → back to Menu from History
        if (kb.escapeKey.wasPressedThisFrame)
        {
            if (HistoryActiveOrZooming()) BackFromHistory();
			else if (!menuVisible)        ToMenu();
			return;
        }

        // Shortcuts only when Menu is visible
        if (menuVisible)
        {
            if (kb.sKey.wasPressedThisFrame)
			{
				// Default
				var d = defaultDifficulty;

				// If you hold a digit while pressing S, pick that difficulty
				bool one  = (kb.digit1Key?.isPressed ?? false) || (kb.numpad1Key?.isPressed ?? false);
				bool two  = (kb.digit2Key?.isPressed ?? false) || (kb.numpad2Key?.isPressed ?? false);
				bool three= (kb.digit3Key?.isPressed ?? false) || (kb.numpad3Key?.isPressed ?? false);

				if      (one)   d = Difficulty.Easy;
				else if (two)   d = Difficulty.Medium;
				else if (three) d = Difficulty.Hard;

				StartGame(d);
			}

			if (kb.hKey.wasPressedThisFrame) ShowHistory();
			}
	}

    // ---------------- actions ----------------        
	public void StartGame() => StartGame(defaultDifficulty);                 
	public void StartGame(Difficulty diff) => StartCoroutine(Co_StartGame(diff)); 
    public void ShowHistory() => StartCoroutine(Co_ShowHistory());

    IEnumerator Co_StartGame(Difficulty diff)
    {
        var btn = GetButtonFor(diff);
		yield return FlashButton(btn, pressedColor, feedbackSeconds);

        // Hide both canvases hard
        SafeSetCanvasVisible(menuCanvas,   false);
        SafeSetCanvasVisible(historyCanvas,false);
        if (historyCanvasGroup)
        {
            historyCanvasGroup.alpha = 0f;
            historyCanvasGroup.interactable = false;
            historyCanvasGroup.blocksRaycasts = false;
        }
        menuVisible = false;

        // Start Python (optional)
        if (arduinoRecorder) arduinoRecorder.StartRecording();
        if (arucoBridge)
		{
			arucoBridge.StartBridge();

			// wait up to ~3 seconds for the process to be live
			float t0 = Time.realtimeSinceStartup;
			yield return new WaitUntil(() =>
				(arucoBridge.IsRunning) || (Time.realtimeSinceStartup - t0 > 3f));

			if (!arucoBridge.IsRunning)
				Debug.LogWarning("[Menu] Arduino bridge failed to start in time.");
		}

        // Enable gameplay and begin the trial
        if (sequence)
        {
			var hm = TrialHistoryManager.Instance;
			sequence.currentTrialNumber = hm ? hm.NextTrialNumber() : 1;
            sequence.enabled = true;
            sequence.BeginTrial();
        }
    }

    IEnumerator Co_ShowHistory()
	{
		yield return FlashButton(historyButton, pressedColor, feedbackSeconds);

		// Hide the menu immediately
		SafeSetCanvasVisible(menuCanvas, false);
		SafeSetCanvasVisible(historyCanvas, false);
		

		// Stop gameplay visuals and hide ALL runtime UI (instructions, books, live scores)
		if (sequence)
		{
			sequence.enabled = false;
			sequence.HideAllUI();            // <- hides blackboard + live Scores + books
			sequence.ResetTablesToOriginal();
		}
    menuVisible = false;

    // Show History UI (your canvas if you use one)
    
    if (historyCanvasGroup)
    {
        historyCanvasGroup.alpha = 0f;
        historyCanvasGroup.interactable = false;
        historyCanvasGroup.blocksRaycasts = false;
    }
   
    if (historyZoomDirector)
    {
        historyZoomDirector.time = 0;
        historyZoomDirector.Play();
    }
    if (historyPresenter) historyPresenter.ShowList();
}
	
    public void BackFromHistory()
    {
        // Rewind Timeline
        if (historyZoomDirector)
        {
            historyZoomDirector.time = 0;
            historyZoomDirector.Stop();
            historyZoomDirector.Evaluate();
        }

        // Hide History
        if (historyCanvasGroup)
        {
            historyCanvasGroup.alpha = 0f;
            historyCanvasGroup.interactable = false;
            historyCanvasGroup.blocksRaycasts = false;
        }
        SafeSetCanvasVisible(historyCanvas, false);

        // Back to Menu
        ToMenu();

        // Stop Python just in case
        if (arucoBridge) arucoBridge.StopPython();
    }

    public void ToMenu()
    {
        StopAllCoroutines();

        // Ensure History is hidden and rewound
        if (historyZoomDirector)
        {
            historyZoomDirector.time = 0;
            historyZoomDirector.Stop();
            historyZoomDirector.Evaluate();
        }
        if (historyCanvasGroup)
        {
            historyCanvasGroup.alpha = 1f;
            historyCanvasGroup.interactable = true;
            historyCanvasGroup.blocksRaycasts = true;
        }
        SafeSetCanvasVisible(historyCanvas, false);

        // Hide gameplay & UI
        if (sequence)
        {
            sequence.HideAllUI();
            sequence.ResetTablesToOriginal();
            sequence.enabled = false;
        }

        // Show Menu (both active and enabled)
        SafeSetCanvasVisible(menuCanvas, true);
        menuVisible = true;

        // Reset button visuals
        SetButtonPalette(easyButton,   easyidleColor);
        SetButtonPalette(mediumButton,   mediumidleColor);
        SetButtonPalette(hardButton,   hardleColor);
        SetButtonPalette(historyButton, historyleColor);
    }

    // ---------------- helpers ----------------
    Button GetButtonFor(Difficulty d)
	{
		switch (d)
		{
			case Difficulty.Easy:   return easyButton;
			case Difficulty.Medium: return mediumButton;
			case Difficulty.Hard:   return hardButton;
			default:                return easyButton;
		}
	}
	
	void HideMenuUI()
	{
		if (!menuCanvas) return;

		// If a CanvasGroup exists on the menu root, make it inert
		var cg = menuCanvas.GetComponent<CanvasGroup>();
		if (cg)
		{
			cg.alpha = 0f;
			cg.interactable = false;
			cg.blocksRaycasts = false;
		}

		// Disable both the Canvas component and the GameObject (belt + suspenders)
		menuCanvas.enabled = false;
		menuCanvas.gameObject.SetActive(false);
	}

	void ShowHistoryUI()
	{
		if (!historyCanvas) return;

		historyCanvas.gameObject.SetActive(true);
		historyCanvas.enabled = true;

		if (historyCanvasGroup)
		{
			historyCanvasGroup.alpha = 1f;
			historyCanvasGroup.interactable = true;
			historyCanvasGroup.blocksRaycasts = true;
		}
	}
    
    
    static void SafeSetCanvasVisible(Canvas c, bool on)
    {
        if (!c) return;
        c.gameObject.SetActive(on);
        c.enabled = on; // avoid active-but-disabled state
    }

    bool HistoryActiveOrZooming()
    {
        bool historyUI =
            (historyCanvas && historyCanvas.isActiveAndEnabled) ||
            (historyCanvasGroup && historyCanvasGroup.alpha > 0.01f);

        bool timelineActive =
            historyZoomDirector &&
            (historyZoomDirector.state == PlayState.Playing || historyZoomDirector.time > 0.01);

        return historyUI || timelineActive;
    }

    void SetButtonPalette(Button btn, Color baseCol)
    {
        if (!btn || !btn.image) return;
        btn.image.color = baseCol;
        var cb = btn.colors;
        cb.normalColor      = baseCol;
        cb.highlightedColor = Color.Lerp(baseCol, Color.white, 0.15f);
        cb.pressedColor     = Color.Lerp(baseCol, Color.black, 0.15f);
        cb.selectedColor    = cb.highlightedColor;
        cb.disabledColor    = new Color(baseCol.r, baseCol.g, baseCol.b, 0.5f);
        btn.colors = cb;
    }

    IEnumerator FlashButton(Button btn, Color flashColor, float seconds)
	{
		if (!btn || !btn.targetGraphic) yield break;

		
		var cb = btn.colors;
		var normalBefore = cb.normalColor;

		// set normal to flash, apply instantly
		cb.normalColor = flashColor;
		btn.colors = cb;
		btn.targetGraphic.CrossFadeColor(flashColor, 0f, true, true);

		yield return new WaitForSecondsRealtime(Mathf.Max(0f, seconds));

		// restore
		cb.normalColor = normalBefore;
		btn.colors = cb;
		btn.targetGraphic.CrossFadeColor(normalBefore, 0f, true, true);
	}
}


