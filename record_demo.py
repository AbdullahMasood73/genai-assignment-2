"""Record a captioned 5-7 minute walkthrough of the running local application.

Requires the app running on http://127.0.0.1:8764 and Playwright (dev-only dependency);
set PLAYWRIGHT_BROWSERS_PATH to a folder containing Playwright's ffmpeg build (ffmpeg-1013).
The WebM video is written under .tools/recording; convert to MP4 with ffmpeg.
"""
from pathlib import Path
from playwright.sync_api import sync_playwright

root=Path(__file__).resolve().parent
folder=root/".tools"/"recording"
folder.mkdir(parents=True,exist_ok=True)
EDGE=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path=EDGE,headless=True)
    ctx=browser.new_context(viewport={"width":1280,"height":860},record_video_dir=str(folder),record_video_size={"width":1280,"height":860},accept_downloads=True)
    page=ctx.new_page()
    page.goto("http://127.0.0.1:8764")
    def caption(text,seconds=11):
        page.evaluate("""text=>{let c=document.getElementById('demo-caption');if(!c){c=document.createElement('aside');c.id='demo-caption';Object.assign(c.style,{pointerEvents:'none',position:'fixed',bottom:'0',left:'0',right:'0',background:'#132e2c',color:'white',padding:'14px 40px',font:'19px/1.45 Segoe UI',zIndex:100});document.body.append(c)}c.textContent=text}""",text)
        page.wait_for_timeout(seconds*1000)
    def scroll(locator):
        # Centre the element so the caption bar at the bottom never hides it.
        locator.evaluate("e=>e.scrollIntoView({block:'center'})"); page.wait_for_timeout(400)
    def run():
        page.locator("#run").click()
        page.wait_for_function("document.getElementById('run').disabled===false",timeout=180000)

    caption("Trial screening workbench - Muhammad Abdullah Masood, I230756. A locally running, fine-tuned Pythia-70M decoder matches patient notes to individual trial criteria. Educational prototype, not a clinical tool.",13)
    caption("The launcher Start-Screening.cmd created the Python environment, verified the pinned base checkpoints by SHA-256, loaded the adapted target and two draft models, and opened this page. The header confirms the loaded checkpoint.",13)
    page.locator("#run").click()
    page.locator(".error").wait_for()
    scroll(page.locator("#status"))
    caption("Input validation: submitting an empty form returns a specific message, shown in red, and the form stays usable for correction.")
    page.locator("#note").fill("0. A 30-year-old woman.\n0. Duplicate number.")
    page.locator("#criteria").fill("inclusion | Age above 18")
    run(); scroll(page.locator("#status"))
    caption("Two sentences share the ID 0, so citations would be ambiguous. The request is rejected with the exact reason. Nothing is silently guessed.",10)
    page.locator("#note").fill("0. A 30-year-old woman.")
    page.locator("#criteria").fill("Age above 18")
    run(); scroll(page.locator("#status"))
    caption("A criterion line without 'inclusion |' or 'exclusion |' is rejected with its line number, because the label set depends on the criterion type.",10)
    page.locator("#example").click()
    caption("Load example inserts an invented note with numbered sentences and five criteria. Numbered lines keep their original IDs so every citation can be traced back to the record.",12)
    page.locator("#mode").select_option("compare")
    caption("Compare all three modes runs uncached, KV-cached and speculative greedy decoding for every criterion, on the same weights and grammar, and checks that all three produce identical tokens.",12)
    run()
    scroll(page.locator("#results section").first)
    caption("The overall outcome is a preliminary aggregation by explicit rules R1 to R5, not a ground-truth label. Missing-information criteria, unsupported decisions and label-evidence inconsistencies are listed for review.",13)
    arts=page.locator("article")
    for i in range(arts.count()):
        scroll(arts.nth(i))
        caption(["Each card shows the criterion type, the predicted label from the type-specific label set, a template explanation and the verbatim cited sentences with their original IDs.",
                 "Latency for each decoding mode is measured on this request. Token agreement confirms that the cached and speculative outputs equal the uncached reference.",
                 "Exclusion criteria use their own label set: excluded, not excluded, not enough information or not applicable. Absent evidence is never treated as proof.",
                 "A label without any cited sentence is flagged as unsupported, so a reviewer knows the decision is not grounded in the note.",
                 "Not enough information marks criteria the record cannot resolve, such as a laboratory value that was never reported."][i%5],11)
    scroll(page.locator("#results section").last)
    caption("The numbered record highlights every sentence cited by at least one criterion, which makes it easy to check whether the evidence actually supports the decision.")
    with page.expect_download() as dl: page.locator("#download").click()
    caption("Export report saves a readable HTML file with the notice, outcome and rules, every criterion with evidence, the numbered record and provenance: model, checkpoint hash, base revision, stop bias, decoding mode and UTC time. JSON export is also available.",14)
    page.locator("#note").fill("The patient is a 62-year-old man with chronic kidney disease. He takes metformin daily. No HbA1c result is available.")
    page.locator("#criteria").fill("inclusion | Age between 18 and 75 years.\ninclusion | HbA1c above 7.5%.\nexclusion | Current insulin therapy.\nexclusion | Pregnancy or breastfeeding.")
    page.locator("#mode").select_option("speculative")
    caption("Now an unseen free-text note. Sentences are split and numbered automatically from zero. This run uses speculative decoding alone.",11)
    run()
    scroll(page.locator("#results section").first)
    caption("Review each decision against the note: the age is stated, HbA1c is explicitly missing, insulin is not mentioned, and a pregnancy criterion may not apply to a male patient. The model can still be wrong; that is why evidence is shown.",15)
    scroll(page.locator("article").first)
    caption("The card reports how many draft tokens the target accepted. Speculation preserves the output exactly; it is only faster when the draft is cheap and agrees often.",12)
    page.locator("nav button[data-tab=lab]").click()
    caption("The Inference lab runs fresh measurements, not stored charts. Choose prompt length, output length, repeats and the draft model.",11)
    page.locator("#plen").fill("512"); page.locator("#olen").fill("24"); page.locator("#reps").fill("3")
    page.locator("#bench").click()
    page.wait_for_function("document.getElementById('bench').disabled===false",timeout=300000)
    scroll(page.locator("#benchOut"))
    caption("After a warm-up, the median of the repeats is reported for prefill, decode, total time, tokens per second, peak memory and analytic KV-cache size. Every path must equal the uncached reference token for token.",15)
    caption("Caching removes the repeated re-encoding of the prompt, so its advantage grows with context length. The smallest top-2 logit margin shows how far the run is from a rounding-induced token flip.",13)
    page.locator("#draftSel").select_option("pythia14m")
    page.locator("#bench").click()
    page.wait_for_function("document.getElementById('bench').disabled===false",timeout=300000)
    scroll(page.locator("#benchOut"))
    caption("The same benchmark with the distilled Pythia-14M draft instead of the early-exit draft. Acceptance rate and draft cost together decide whether speculation helps on this CPU.",13)
    page.locator("#aq").fill("256"); page.locator("#ak").fill("256")
    page.locator("#attn").click()
    page.wait_for_function("document.getElementById('attn').disabled===false",timeout=120000)
    scroll(page.locator("#attnOut"))
    caption("The attention check compares the reference softmax, the tiled online-softmax kernel and PyTorch's fused kernel against an FP64 ground truth. The tiled kernel is exact to about 1e-7 but slower in Python, while storing only one score tile.",15)
    scroll(page.locator("#logs"))
    caption("Logged results from the report: held-out test metrics on the eight test patients and the controlled decoding benchmark. Fine-tuning clearly beats the unadapted model, but on accuracy it is statistically tied with a majority-label rule, and its evidence is close to always citing the first sentence.",16)
    page.locator("nav button[data-tab=about]").click()
    caption("Rules and provenance documents the aggregation rules, label meanings, loaded model details and limitations.",11)
    page.locator("nav button[data-tab=screen]").click()
    page.locator("#criteria").fill("inclusion | " + "very long rule " * 200)
    run(); scroll(page.locator("#status"))
    caption("Recovery from invalid input: an over-long criterion is rejected with a clear message and the server keeps running.",11)
    caption("Data: the TrialGPT criterion annotations with the prescribed 37/8/8 patient split and seed 42. Only the note, criterion type and text enter the model; GPT-4 predictions are excluded and leakage is tested.",14)
    caption("Thank you. Code, logs, split manifest, weights and the IEEE report are in the repository and the submission archive. Every prediction requires independent professional review.",12)
    ctx.close()
    print(page.video.path())
    browser.close()
