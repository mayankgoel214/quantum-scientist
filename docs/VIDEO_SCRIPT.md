# Demo video voiceover: 1 minute 33 seconds

The silent video is on your Desktop: **`q-scientist-video-silent.mp4`**.

**How to record (about 5 minutes):**
1. Open the silent video in QuickTime and put it on one side of the screen.
2. Open the voice recorder (QuickTime: File → New Audio Recording, or Voice Memos on your phone).
3. Start the recording, then press play on the video, and read each block **when its time comes up**.
   Talk at a normal, calm pace. Pauses are fine. If you finish a block early, just wait for the next one.
4. Send me the audio file. I'll line it up, clean the volume, and give you the final video.

---

**0:00 – 0:09 · Title card**
> Last night, an AI ran experiments on IBM's real quantum computer while I slept. This is Q-Scientist.

**0:09 – 0:21 · The live site, showing "24"**
> The question: how many qubits can this 156-qubit chip genuinely entangle? The agent's answer, on real hardware: twenty-four, statistically certified.

**0:21 – 0:52 · The replay plays, then the results light up** (longest block, take your time)
> Here's the night replayed. Gemini read the chip's live calibration and designed eight experiments. Before sending anything, it predicted every result with a digital twin, a noise simulator I built, and froze those predictions. It picked which qubits to use, avoiding the noisy ones in red, and submitted through Qiskit Runtime. The green tree is the real 24-qubit entangled state on the chip, and it passed the certification test.

**0:52 – 1:05 · Prediction vs hardware chart**
> Because the predictions were frozen first, I can check them honestly. The twin called certification right both times, but its error doubled as the states grew: noise the calibration data misses.

**1:05 – 1:24 · Scrolling the paper**
> Then the agent wrote this paper on its own. Methods, figures, confidence intervals, limitations, and IBM job IDs so anyone can verify it. Every number is computed from the run log, never typed by the AI.

**1:24 – 1:33 · End card**
> Q-Scientist: an AI scientist that keeps itself honest. The live site and code are linked below. Thanks!
