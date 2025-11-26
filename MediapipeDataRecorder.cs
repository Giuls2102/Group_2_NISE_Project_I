using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;

public class MediapipeDataRecorder : MonoBehaviour
{
    [Header("Control")]
    public KeyCode startKey = KeyCode.S;   // press S to start

    [Header("Output")]
    public string filePrefix = "mediapipe_trial_";

    private bool isRecording = false;
    private float trialStartTime;
    private StreamWriter writer;

    void Update()
    {
        // Only START on S, never stop with a key
        if (Input.GetKeyDown(startKey) && !isRecording)
        {
            StartRecording();
        }
    }

    public void StartRecording()
    {
        if (isRecording)
            return;

        isRecording = true;
        trialStartTime = Time.time;

        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string fileName = filePrefix + timestamp + ".csv";
        string path = Path.Combine(Application.persistentDataPath, fileName);

        writer = new StreamWriter(path);
        writer.AutoFlush = true;

        // Adjust to match your landmarks data
        writer.WriteLine("time,landmarkIndex,x,y,z");

        Debug.Log("Mediapipe recording STARTED: " + path);
    }

    public void StopRecording()
    {
        if (!isRecording)
            return;

        isRecording = false;

        if (writer != null)
        {
            writer.Flush();
            writer.Close();
            writer = null;
        }

        Debug.Log("Mediapipe recording STOPPED.");
    }

    private void OnDestroy()
    {
        StopRecording();
    }

    /// <summary>
    /// Call this from your Mediapipe script every time you get landmarks.
    /// </summary>
    public void RecordLandmarks(IReadOnlyList<Vector3> landmarks)
    {
        if (!isRecording || writer == null || landmarks == null)
            return;

        float t = Time.time - trialStartTime;

        for (int i = 0; i < landmarks.Count; i++)
        {
            Vector3 p = landmarks[i];
            writer.WriteLine($"{t:F4},{i},{p.x:F6},{p.y:F6},{p.z:F6}");
        }
    }

    /// <summary>
    /// Call this when the trial ends
    /// (e.g. when the last book is placed and Arduino stops).
    /// </summary>
    public void OnTrialEnd()
    {
        StopRecording();
    }
}
