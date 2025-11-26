using System.Text;
using TMPro;
using UnityEngine;

public class HistoryList : MonoBehaviour
{
    [Header("Pink board TMP_Text")]
    [SerializeField] TMP_Text target;

    [Header("Import from Python file on open")]
    [SerializeField] bool importOnShow = true;

    public void ShowList()
    {
        if (!target) return;

        var hm = TrialHistoryManager.Instance;
        if (hm == null)
        {
            target.text = "History\n(no manager)";
            target.gameObject.SetActive(true);
            return;
        }

        // Pull latest Python JSON → append a new trial (idempotent by content hash)
        if (importOnShow) hm.ImportFromPythonJsonAsNewTrial();

        if (hm.All == null || hm.All.Count == 0)
        {
            target.text = "History\nNo trials yet.";
            target.gameObject.SetActive(true);
            return;
        }


        var sb = new StringBuilder();
        sb.AppendLine("Verlauf");
        sb.AppendLine();

        foreach (var r in hm.All)
        {
            sb.AppendLine($"Durchgang {r.trial}: Versuche {r.attempts} - Zeit {r.seconds:0.00}s - Genauigkeit {r.accuracy:0.0}%");
            bool anyBooks = (r.timeBook1 >= 0f) || (r.timeBook2 >= 0f) || (r.timeBook3 >= 0f);
            if (anyBooks)
            {
                sb.Append("   ");
                if (r.timeBook1 >= 0f) sb.Append($"Buch1:{r.timeBook1:0.00}s  ");
                if (r.timeBook2 >= 0f) sb.Append($"Buch2:{r.timeBook2:0.00}s  ");
                if (r.timeBook3 >= 0f) sb.Append($"Buch3:{r.timeBook3:0.00}s");
                sb.AppendLine();
            }
        }

        target.text = sb.ToString();
        target.gameObject.SetActive(true);
    }
}
