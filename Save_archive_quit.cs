using UnityEngine;
using System;
using System.IO;

public class SaveArchiveAndCleanupOnQuit : MonoBehaviour
{
    [Tooltip("Create a timestamped copy of trial_history.json on quit")]
    public bool archiveHistory = true;

    [Tooltip("If empty, archives next to trial_history.json")]
    public string archiveFolder = "";

    [Tooltip("Delete Python’s trial_times.json on quit")]
    public bool deletePythonTimes = true;

    bool _didRun = false; // guard so we do it once

    void OnApplicationQuit()   { RunOnce(); }
    void OnDisable()           { if (Application.isPlaying) RunOnce(); }
    void OnDestroy()           { if (Application.isPlaying) RunOnce(); }

    void RunOnce()
    {
        if (_didRun) return;
        _didRun = true;

        var hm = TrialHistoryManager.Instance;
        if (hm == null) return;

        // 1) Save latest Unity history
        hm.Save();

        // 2) Optional archive (timestamped copy)
        if (archiveHistory) ArchiveHistory(hm);

        // 3) Optional cleanup of trial_times.json
        if (deletePythonTimes) DeletePythonTimes(hm);
    }

    void ArchiveHistory(TrialHistoryManager hm)
    {
        try
        {
            string src = hm.GetHistorySavePath();
            if (!File.Exists(src)) return;

            string folder = string.IsNullOrEmpty(archiveFolder) ? Path.GetDirectoryName(src) : archiveFolder;
            Directory.CreateDirectory(folder);

            string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string dst = Path.Combine(folder, $"trial_history_{stamp}.json");

            File.Copy(src, dst, overwrite: true);
            Debug.Log($"[QuitSave] Archived history to: {dst}");
        }
        catch (Exception e)
        {
            Debug.LogWarning("[QuitSave] Archive failed: " + e.Message);
        }
    }

    void DeletePythonTimes(TrialHistoryManager hm)
    {
        try
        {
            string path = hm.ResolvePythonTimesPath(); // exact file TrialHistoryManager imports
            if (!string.IsNullOrEmpty(path) && File.Exists(path))
            {
                File.Delete(path);
                Debug.Log("[QuitSave] Deleted Python times file: " + path);
            }
            else
            {
                Debug.Log("[QuitSave] Python times file not found to delete: " + (path ?? "(null)"));
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning("[QuitSave] Delete failed: " + e.Message);
        }
    }
}
