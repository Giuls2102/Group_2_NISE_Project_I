using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Globalization;
using System.Text.RegularExpressions;
using UnityEngine;
using Debug = UnityEngine.Debug;
using Object = UnityEngine.Object;

public class PythonArucoBridge : MonoBehaviour
{
    [Header("Python process")]
    public string pythonPath;     // /Users/giulia/py312-unity/bin/python3
    public string scriptPath;     // .../three_zone_aruco.py
    [TextArea] public string extraArgs;

    [Header("Environment (mac: add Homebrew bins so ffmpeg is found)")]
    public string prependToPATH = "/opt/homebrew/bin:/usr/local/bin";

    [Header("Route events into the game (optional)")]
    public Behaviour sequence;   // SequenceTablesFeedback on a GameObject
    struct ArucoEvt { public bool ok; public string obj; public string zone; public string expected; }
    readonly Queue<ArucoEvt> _pending = new Queue<ArucoEvt>();
    readonly object _lock = new object();
    SequenceTablesFeedback _seq;

    // NEW: queue for time events (parsed from python stdout)
    struct TimeEvt { public string obj; public float secs; }
    readonly Queue<TimeEvt> _pendingTimes = new Queue<TimeEvt>();
    readonly object _lockTimes = new object();

    [Header("History capture")]
    [SerializeField] TrialHistoryManager history;         // auto-found if left empty
    [SerializeField] public int currentTrialNumber = 1;   // set from Menu or auto

    [Header("When to auto-start/stop")]
    public bool startOnEnable = false;
    public bool stopOnDisable = false;

    [Header("Camera selection (mac/avfoundation)")]
    [Tooltip("Substrings to prefer in the ffmpeg device list (checked in order).")]
    public string[] preferredDeviceSubstrings = new[]
    {
        "VID:1133",
        "USB Camera"
    };
    [Tooltip("Substrings to exclude (case-insensitive).")]
    public string[] excludeDeviceSubstrings = new[]
    {
        "FaceTime"
    };
    [Tooltip("Resolution hint only for logging; actual selection is in Python script after we patch AVF_DEVICE.")]
    public string forcedResHint = "1280x720";

    static readonly string[] FfmpegCandidates =
    {
        "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "ffmpeg"
    };

    Process proc;
    bool isStarting = false;
    bool isStopping = false;
    

    public bool IsRunning
    {
        get { try { return proc != null && !proc.HasExited; } catch { return false; } }
    }
    public void StartBridge() => StartPython();
    public void StopBridge()  => StopPython();

    void OnEnable()
    {
        if (startOnEnable) StartPython();
    }

    void OnDisable()
    {
        if (stopOnDisable) StopPython();
    }

    void OnApplicationQuit()
    {
        StopPython();
    }

    public void StartPython()
    {
        if (proc != null && !proc.HasExited)
        {
            Debug.Log("[ArucoBridge] Already running.");
            return;
        }
        if (isStarting) return;
        StartCoroutine(Co_StartPython());
    }

    public void StopPython()
    {
        if (isStopping) return;
        StartCoroutine(Co_StopPython());
    }

    IEnumerator Co_StartPython()
    {
        isStarting = true;

        if (string.IsNullOrEmpty(pythonPath) || string.IsNullOrEmpty(scriptPath))
        {
            Debug.LogWarning("[ArucoBridge] pythonPath or scriptPath not set.");
            isStarting = false;
            yield break;
        }

        string ffmpegExe = GetFfmpegExecutable();
        if (string.IsNullOrEmpty(ffmpegExe))
        {
            Debug.LogError("[ArucoBridge] ffmpeg not found. Install with Homebrew and set PATH.");
            isStarting = false;
            yield break;
        }

        string deviceList = GetFfmpegDeviceList(ffmpegExe);
        if (string.IsNullOrWhiteSpace(deviceList))
        {
            Debug.LogError("[ArucoBridge] Could not get ffmpeg device list.");
            isStarting = false;
            yield break;
        }
        Debug.Log($"[ArucoBridge] Device list:\n{deviceList}");

        int avIndex = ResolveExternalVideoIndex(deviceList, preferredDeviceSubstrings, excludeDeviceSubstrings);
        if (avIndex < 0)
        {
            Debug.LogError("[ArucoBridge] External camera NOT found.");
            isStarting = false;
            yield break;
        }
        Debug.Log($"[ArucoBridge] Using external camera index {avIndex} (avfoundation), res hint {forcedResHint}");

        string patchedScript = CreatePatchedScript(scriptPath, avIndex);
        if (string.IsNullOrEmpty(patchedScript))
        {
            Debug.LogError("[ArucoBridge] Failed to create patched script.");
            isStarting = false;
            yield break;
        }

        var psi = new ProcessStartInfo
        {
            FileName = pythonPath,
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        if (!string.IsNullOrEmpty(prependToPATH))
        {
            string existingPath = Environment.GetEnvironmentVariable("PATH") ?? "";
            psi.Environment["PATH"] = prependToPATH + ":" + existingPath;
        }

        psi.ArgumentList.Clear();
        psi.ArgumentList.Add(patchedScript);
        if (!string.IsNullOrWhiteSpace(extraArgs))
        {
            foreach (var part in extraArgs.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries))
                psi.ArgumentList.Add(part);
        }

        try
        {
            proc = new Process();
            var outDir = System.IO.Path.Combine(Application.temporaryCachePath, "aruco");
			System.IO.Directory.CreateDirectory(outDir);               
			// in PythonArucoBridge, before proc.Start()
			psi.Environment["OUTPUT_DIR"]  = outDir;
			psi.Environment["TIMES_ALIAS"] = "trial_history.json";
            proc.StartInfo = psi;

            proc.OutputDataReceived += (s, e) =>
            {
                if (string.IsNullOrEmpty(e.Data)) return;
                var line = e.Data.Trim();
                Debug.Log($"[ArucoBridge:stdout] {line}");

                // ------------- PARSING -------------
                // Normalize
                var lower = line.ToLowerInvariant();

                // Extract object name: Book1/Book2/Book3
                string objName = ExtractBookName(lower); // returns "Book1" / "Book2" / "Book3" or null
				// Example: "Book1: 1.23s" or "Book2: 0.98 s"
				
                // Extract time if present: "time_s=1.23"
                if (TryParseTimeSeconds(lower, out float secs) && !string.IsNullOrEmpty(objName))
                {
                    lock (_lockTimes) _pendingTimes.Enqueue(new TimeEvt { obj = objName, secs = secs });
                }
                
                

                // Simple zone parsing helpers (for game feedback events)
                string AfterToken(string src, string token)
                {
                    int i = src.IndexOf(token, StringComparison.OrdinalIgnoreCase);
                    if (i < 0) return null;
                    i += token.Length;
                    var sb = new System.Text.StringBuilder();
                    while (i < src.Length && char.IsLetter(src[i])) { sb.Append(src[i]); i++; }
                    return sb.Length > 0 ? sb.ToString().ToLowerInvariant() : null;
                }

                // Dispatch OK/ERR events to Sequence (if present)
                if (lower.Contains(" correct"))
                {
                    string zone = AfterToken(lower, " in ");
                    var ev = new ArucoEvt { ok = true, obj = objName ?? "", zone = zone ?? "", expected = "" };
                    lock (_lock) _pending.Enqueue(ev);
                    return;
                }
                if (lower.Contains(" incorrect"))
                {
                    string wrongZone = AfterToken(lower, " in ");
                    string expected  = AfterToken(lower, " to ");
                    var ev = new ArucoEvt { ok = false, obj = objName ?? "", zone = wrongZone ?? "", expected = expected ?? "" };
                    lock (_lock) _pending.Enqueue(ev);
                    return;
                }

                if (line == "n")
                {
                    var ev = new ArucoEvt { ok = false, obj = objName ?? "", zone = "", expected = "" };
                    lock (_lock) _pending.Enqueue(ev);
                }
            };

            proc.ErrorDataReceived += (s, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data)) Debug.LogWarning($"[ArucoBridge:stderr] {e.Data}");
            };
			
            if (!proc.Start())
            {
                Debug.LogError("[ArucoBridge] Failed to start process.");
                proc = null;
                isStarting = false;
                yield break;
            }

            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();

            Debug.Log($"[ArucoBridge] Started: {pythonPath} \"{patchedScript}\"");
        }
        catch (Exception ex)
        {
            Debug.LogError($"[ArucoBridge] Exception starting process: {ex}");
            proc = null;
        }

        isStarting = false;
    }

    void Update()
    {
        // Find Sequence component once
        if (_seq == null)
        {
            if (sequence) _seq = sequence.GetComponent<SequenceTablesFeedback>();
            if (_seq == null)
            {
#if UNITY_2023_1_OR_NEWER
                _seq = Object.FindAnyObjectByType<SequenceTablesFeedback>();
#else
                _seq = Object.FindObjectOfType<SequenceTablesFeedback>();
#endif
                // don't early-return; we still want to drain time events to history even without Sequence
            }
        }

        // Drain game feedback events (correct / wrong)
        while (true)
        {
            ArucoEvt ev;
            lock (_lock)
            {
                if (_pending.Count == 0) break;
                ev = _pending.Dequeue();
            }

            Debug.Log($"[ArucoBridge] Dispatch {(ev.ok ? "OK" : "ERR")} obj='{ev.obj}' zone='{ev.zone}' expected='{ev.expected}'");

            if (_seq != null)
            {
                if (ev.ok) _seq.OnArucoCorrect(ev.obj, ev.zone);
                else       _seq.OnArucoWrong(ev.obj, ev.zone, ev.expected);
            }
        }

        // Drain parsed per-book times → History (main thread)
        while (true)
        {
            TimeEvt te;
            lock (_lockTimes)
            {
                if (_pendingTimes.Count == 0) break;
                te = _pendingTimes.Dequeue();
            }

            var hm = history ? history : TrialHistoryManager.Instance;
            if (hm == null) continue;

            if (currentTrialNumber <= 0) currentTrialNumber = hm.NextTrialNumber();

            // Writes straight into your JSON-managed history file
            //hm.SetBookTime(currentTrialNumber, te.obj, te.secs);
            // Debug.Log($"[ArucoBridge] Saved {te.obj}={te.secs:0.00}s to trial {currentTrialNumber}");
        }
    }

    IEnumerator Co_StopPython()
    {
        isStopping = true;

        if (proc == null) { isStopping = false; yield break; }
        if (proc.HasExited) { proc.Dispose(); proc = null; isStopping = false; yield break; }

        try { proc.StandardInput.WriteLine("q"); proc.StandardInput.Flush(); } catch { }

        float t0 = Time.realtimeSinceStartup;
        while (!proc.HasExited && Time.realtimeSinceStartup - t0 < 2f)
            yield return null;

        if (!proc.HasExited)
        {
            try { proc.Kill(); } catch { }
        }
        proc.Dispose();
        proc = null;
        isStopping = false;
        Debug.Log("[ArucoBridge] Python exited.");
    }

    // ---------------- parsing helpers ----------------

    // Finds "book1"/"book2"/"book3" in the line, returns "Book1"/"Book2"/"Book3"
    static string ExtractBookName(string lower)
    {
        if (lower.Contains("book1")) return "Book1";
        if (lower.Contains("book2")) return "Book2";
        if (lower.Contains("book3")) return "Book3";
        return null;
    }
    

    // Matches "time_s=1.23" (any float)
    static readonly Regex rxTime = new Regex(@"time_s\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);
	
	
    static bool TryParseTimeSeconds(string lower, out float secs)
    {
        var m = rxTime.Match(lower);
        if (m.Success && float.TryParse(m.Groups[1].Value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out secs))
            return true;
        secs = 0f;
        return false;
    }

    // ---------------- ffmpeg helpers ----------------

    string GetFfmpegExecutable()
    {
        foreach (var cand in FfmpegCandidates)
        {
            try
            {
                var p = Process.Start(new ProcessStartInfo
                {
                    FileName = cand,
                    Arguments = "-version",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                });
                if (p != null)
                {
                    p.WaitForExit(500);
                    p.Dispose();
                    return cand;
                }
            }
            catch { }
        }
        return null;
    }

    string GetFfmpegDeviceList(string ffmpegExe)
    {
        return RunAndCapture(ffmpegExe, "-f avfoundation -list_devices true -i \"\"");
    }

    int ResolveExternalVideoIndex(string ffmpegList, string[] prefers, string[] excludes)
    {
        var lines = ffmpegList.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
        var candidates = new List<(int idx, string name)>();
        var re = new Regex(@"\[(\d+)\]\s+(.+)$");

        foreach (var raw in lines)
        {
            if (!raw.Contains("] [")) continue;
            var m = re.Match(raw);
            if (!m.Success) continue;
            int idx = int.Parse(m.Groups[1].Value);
            string name = m.Groups[2].Value.Trim();

            string lower = name.ToLowerInvariant();
            if (lower.Contains("microfon") || lower.Contains("audio")) continue;

            candidates.Add((idx, name));
        }

        if (candidates.Count == 0) return -1;

        if (excludes != null && excludes.Length > 0)
        {
            candidates.RemoveAll(c =>
            {
                foreach (var ex in excludes)
                    if (!string.IsNullOrEmpty(ex) && c.name.IndexOf(ex, StringComparison.OrdinalIgnoreCase) >= 0)
                        return true;
                return false;
            });
        }

        if (prefers != null)
        {
            foreach (var want in prefers)
            {
                if (string.IsNullOrEmpty(want)) continue;
                foreach (var c in candidates)
                    if (c.name.IndexOf(want, StringComparison.OrdinalIgnoreCase) >= 0)
                        return c.idx;
            }
        }

        int best = -1, maxIdx = -1;
        foreach (var c in candidates)
            if (c.idx > maxIdx) { maxIdx = c.idx; best = c.idx; }
        return best;
    }

    string CreatePatchedScript(string originalPath, int avIndex)
    {
        try
        {
            var src = File.ReadAllText(originalPath);
            var rxLine = new Regex(@"(?m)^\s*AVF_DEVICE\s*=\s*""[^""]*""\s*$");
            string replacementLine = $"AVF_DEVICE = \"{avIndex}:\"";

            string dstSrc = rxLine.IsMatch(src)
                ? rxLine.Replace(src, replacementLine)
                : src.Replace("AVF_DEVICE = \"0:\"", replacementLine);

            string tmpDir = Path.Combine(Application.temporaryCachePath, "aruco");
            Directory.CreateDirectory(tmpDir);
            string tmpFile = Path.Combine(tmpDir, $"three_zone_aruco_{avIndex}.py");
            File.WriteAllText(tmpFile, dstSrc, new UTF8Encoding(false));
            return tmpFile;
        }
        catch (Exception ex)
        {
            Debug.LogError($"[ArucoBridge] Patch script failed: {ex}");
            return null;
        }
    }

    string RunAndCapture(string exe, string args)
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = exe,
                Arguments = args,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            if (!string.IsNullOrEmpty(prependToPATH))
            {
                string existingPath = Environment.GetEnvironmentVariable("PATH") ?? "";
                psi.Environment["PATH"] = prependToPATH + ":" + existingPath;
            }

            using (var p = Process.Start(psi))
            {
                var sb = new StringBuilder();
                p.OutputDataReceived += (s, e) => { if (e.Data != null) sb.AppendLine(e.Data); };
                p.ErrorDataReceived  += (s, e) => { if (e.Data != null) sb.AppendLine(e.Data); };
                p.BeginOutputReadLine();
                p.BeginErrorReadLine();
                p.WaitForExit(4000);
                return sb.ToString();
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[ArucoBridge] RunAndCapture error for '{exe}': {ex.Message}");
            return null;
        }
    }
}
