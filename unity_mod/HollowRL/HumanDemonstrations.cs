using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using InControl;
using UnityEngine;

// Recording is opt-in and keeps every rendered frame in a bounded queue.
// The Python consumer drains batches; short human taps are not lost between
// Python polls. No button-to-macro approximation is made here.
public partial class HollowRL
{
    private readonly object demoGate = new object();
    private readonly Queue<string> demoFrames = new Queue<string>();
    private TcpClient demoOwner;
    private bool demoActive;
    private int demoWarmup;
    private int demoDropped;
    private const int DemoQueueCapacity = 3600;

    private bool DemonstrationActive
    {
        get { lock (demoGate) return demoActive; }
    }

    private bool HandleDemonstrationCommand(TcpClient client, string command, StreamWriter writer)
    {
        if (!command.StartsWith("DEMO ", StringComparison.OrdinalIgnoreCase))
            return false;
        string verb = command.Substring(5).Trim().ToUpperInvariant();
        if (verb == "STATUS")
        {
            writer.WriteLine("demo_v1 " + (DemonstrationActive ? "recording" : "idle"));
            return true;
        }
        if (verb == "START")
        {
            lock (demoGate)
            {
                if (demoOwner != null && demoOwner != client)
                {
                    writer.WriteLine("err another recorder owns this game");
                    return true;
                }
                demoOwner = client;
                demoFrames.Clear();
                demoDropped = 0;
                demoWarmup = 2;  // let injected holds and their release edges clear
                InputInjector.ReleaseAll();
                demoActive = true;
            }
            writer.WriteLine("ok demo_v1 started");
            return true;
        }
        if (verb == "STOP")
        {
            lock (demoGate)
            {
                if (demoOwner != client)
                {
                    writer.WriteLine("err not the recording owner");
                    return true;
                }
                demoActive = false;
            }
            writer.WriteLine("ok demo_v1 stopped");
            return true;
        }
        if (verb == "READ")
        {
            string[] batch;
            int dropped;
            bool active;
            lock (demoGate)
            {
                if (demoOwner != client)
                {
                    writer.WriteLine("err not the recording owner");
                    return true;
                }
                batch = demoFrames.ToArray();
                demoFrames.Clear();
                dropped = demoDropped;
                active = demoActive;
            }
            writer.WriteLine("demo_v1 " + batch.Length + " " + dropped + " " + (active ? 1 : 0));
            foreach (string frame in batch)
                writer.WriteLine(frame);
            return true;
        }
        writer.WriteLine("err unknown demonstration command");
        return true;
    }

    private void StopDemonstrationOnDisconnect(TcpClient client)
    {
        lock (demoGate)
        {
            if (demoOwner == client)
            {
                demoActive = false;
                demoOwner = null;
            }
        }
    }

    private static void RecordButton(PlayerAction action, int bit,
                                     ref int held, ref int pressed, ref int released)
    {
        if (action.IsPressed) held |= bit;
        if (action.WasPressed) pressed |= bit;
        if (action.WasReleased) released |= bit;
    }

    private void LateUpdate()
    {
        lock (demoGate)
        {
            if (!demoActive) return;
            if (demoWarmup > 0) { demoWarmup--; return; }
        }
        InputHandler handler = InputHandler.Instance;
        if (handler == null || handler.inputActions == null) return;
        HeroActions a = handler.inputActions;
        int held = 0, pressed = 0, released = 0;
        // These bit positions are the human_demo.py BUTTON_NAMES schema.
        RecordButton(a.left,       1 << 0, ref held, ref pressed, ref released);
        RecordButton(a.right,      1 << 1, ref held, ref pressed, ref released);
        RecordButton(a.up,         1 << 2, ref held, ref pressed, ref released);
        RecordButton(a.down,       1 << 3, ref held, ref pressed, ref released);
        RecordButton(a.attack,     1 << 4, ref held, ref pressed, ref released);
        RecordButton(a.jump,       1 << 5, ref held, ref pressed, ref released);
        RecordButton(a.dash,       1 << 6, ref held, ref pressed, ref released);
        RecordButton(a.cast,       1 << 7, ref held, ref pressed, ref released);
        RecordButton(a.quickCast,  1 << 8, ref held, ref pressed, ref released);
        RecordButton(a.dreamNail,  1 << 9, ref held, ref pressed, ref released);
        RecordButton(a.superDash,  1 << 10, ref held, ref pressed, ref released);

        // Geometry is the Update snapshot from this same Unity frame; buttons
        // are read after InputHandler.Update. Keep frame/time for precise
        // offline alignment instead of assuming Python's poll time is game time.
        string line = Time.frameCount + "\t" + FormatFloat(Time.unscaledTime)
            + "\t" + FormatFloat(Time.time) + "\t" + held + "\t" + pressed
            + "\t" + released + "\t" + (a.attack.EnabledInHierarchy ? 1 : 0)
            + "\t" + BuildStateLine(TakeSnapshot());
        lock (demoGate)
        {
            if (!demoActive) return;
            if (demoFrames.Count >= DemoQueueCapacity)
            {
                demoFrames.Dequeue();
                demoDropped++;
            }
            demoFrames.Enqueue(line);
        }
    }
}
