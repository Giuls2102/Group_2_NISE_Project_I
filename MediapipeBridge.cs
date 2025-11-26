using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class MediapipeBridge : MonoBehaviour
{
    [Header("UDP settings")]
    public string pythonHost = "127.0.0.1";
    public int pythonPort = 5007;        // choose a free port

    UdpClient client;
    IPEndPoint endPoint;

    void Awake()
    {
        endPoint = new IPEndPoint(IPAddress.Parse(pythonHost), pythonPort);
        client = new UdpClient();
        Debug.Log($"[MediapipeBridge] Sending to {pythonHost}:{pythonPort}");
    }

    void OnDestroy()
    {
        if (client != null)
        {
            client.Close();
            client = null;
        }
    }

    void Send(string msg)
    {
        if (client == null) return;
        var data = Encoding.UTF8.GetBytes(msg);
        client.Send(data, data.Length, endPoint);
        // optional debug:
        // Debug.Log("[MediapipeBridge] Sent: " + msg);
    }

    public void StartRecording(int trialNumber)
    {
        // e.g. "START 3"
        Send($"START {trialNumber}");
    }

    public void OnTrialEnd()
    {
        // e.g. "STOP"
        Send("STOP");
    }
}
