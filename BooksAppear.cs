using System.Collections;
using UnityEngine;
using UnityEngine.InputSystem;

public class BooksAppear : MonoBehaviour
{
    [Header("Book roots to toggle")]
    public GameObject[] books;

    [Header("Initial state & optional key toggle")]
    public bool startHidden = true;
    public bool enableKeyToggle = false;
    public Key triggerKey = Key.S;

    [Header("Delay before showing after Arm()")]
    public float appearDelay = 0.25f;

    bool ready = false;
    Coroutine pending;

    void Awake()
    {
        if (startHidden) SetActive(false);
    }

    void Update()
    {
        if (!enableKeyToggle || Keyboard.current == null) return;

        if (Keyboard.current[triggerKey].wasPressedThisFrame)
        {
            if (ready) Disarm();   // hide immediately
            else Arm();            // will appear after delay
        }
    }

    // Call this when the menu hides / trial begins
    public void Arm()
    {
        ready = true;
        if (pending != null) StopCoroutine(pending);
        pending = StartCoroutine(ShowAfterDelay());
    }

    // Call this when returning to the menu
    public void Disarm()
    {
        ready = false;
        if (pending != null) { StopCoroutine(pending); pending = null; }
        SetActive(false);
    }

    IEnumerator ShowAfterDelay()
    {
        yield return new WaitForSecondsRealtime(Mathf.Max(0f, appearDelay));
        if (ready) SetActive(true);
        pending = null;
    }

    void SetActive(bool on)
    {
        foreach (var b in books) if (b) b.SetActive(on);
    }
}

