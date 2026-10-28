# CC0 / public-domain speech corpora for voice-reference clips

Scope: CC0 or public-domain speech corpora with many distinct speakers, downloadable without login or API key. Freesound excluded. LibriVox counted as already covered.

## Key Question 1: Which CC0 or public-domain speech datasets are downloadable without login?

### Takeaway
Only one verified CC0 multi-speaker speech source surfaced with a confirmed license statement: CoVoST (translation annotations over Common Voice, CC0). Common Voice itself is CC0 per multiple sources, but the official download route's login status was NOT verified. Most other candidates are CC BY, CC BY-SA, CC BY-NC, or unclear. No new CC0 corpus with per-speaker long-form speech was confirmed.

### Cited Findings
- CoVoST: "We've released the data under CC0 license." 78K speakers, 2,880 hours of speech; download instructions point to the GitHub repo and the Common Voice datasets page; no login mentioned — [Meta AI CoVoST page](https://ai.meta.com/tools/covost/)
- CoVoST GitHub README license section: CoVoST data "CC0"; Tatoeba sentences "CC BY 2.0 FR"; Tatoeba speeches "Various CC licenses (please check out the audio_license column)"; anything else "CC BY-NC 4.0" — [facebookresearch/covost](https://github.com/facebookresearch/covost)
- CoVoST download routes are direct HTTPS tar.gz TSVs at dl.fbaipublicfiles.com; the Common Voice audio itself is downloaded separately from the Common Voice datasets page — [facebookresearch/covost](https://github.com/facebookresearch/covost)
- Common Voice license: Wikipedia lists the project as CC0 and describes the sentence database as public domain — [Common Voice (Wikipedia)](https://en.wikipedia.org/wiki/Common_Voice). Mozilla's 2.0 release described as CC0 "No Rights Reserved" per a Mycroft community post — [Mycroft forum](https://community.mycroft.ai/t/mozilla-releases-the-common-voice-2-0-dataset/6090)
- Common Voice code repo is MPL 2.0; the data is CC0 — [common-voice GitHub](https://github.com/common-voice/common-voice) (search snippet, not fetched)
- Common Voice 16.0 community mirror on Hugging Face (fsicoli/common_voice_16_0): license tag `cc0-1.0`, card text "Public Domain, CC-0", no gating notice, `load_dataset` needs no auth step — [HF card](https://huggingface.co/datasets/fsicoli/common_voice_16_0). This is a community re-upload, not the official Mozilla route.
- Thorsten-Voice 2021.02 on Zenodo: CC0, but a single speaker (>23 h) — search snippet listing [Mozilla DataCollective entry](https://datacollective.mozillafoundation.org/datasets/cmm2fwi490024nt0779vv9o0d) and Zenodo context. Single speaker: rejected for this objective.
- VCTK on Hugging Face (vt57299/vctk and CSTR-Edinburgh/vctk): about 44 h, 110 speakers, ~400 sentences per speaker. License NOT confirmed from snippets — [CSTR-Edinburgh/vctk](https://huggingface.co/datasets/CSTR-Edinburgh/vctk). Viewer disabled (loading script). Not counted as CC0 until the card is checked.
- Hi-Fi TTS (OpenSLR 109): about 291.6 h, 10 speakers, CC BY 4.0 — [OpenSLR 109](https://openslr.org/109). Rejected (CC-BY).
- LibriTTS (OpenSLR 60): about 585 h, multi-speaker, derived from LibriSpeech, CC BY 4.0 — [OpenSLR 60](https://www.openslr.org/60). Rejected (CC-BY); LibriVox-derived anyway.
- Speech Wikimedia: 1780 h, 77 languages, CC BY-SA — [arXiv 2308.15710](https://arxiv.org/pdf/2308.15710). Rejected (CC-BY-SA).
- Tatoeba audio: per-recording license chosen by contributor (CC BY, CC BY-SA, CC BY-NC, or none); CC0 not allowed on audio per a 2019-era contributor guide — [Tatoeba downloads](https://tatoeba.org/ca/downloads), [CC0 contributions wiki](https://en.wiki.tatoeba.org/articles/show/cc0-contributions). Rejected for bulk use; a per-file filter could salvage a few CC0-labeled clips, but this was not verified.
- VoxForge: the project says submitted audio is released under the GPL (Google Code archive lists GNU GPL v3) — [VoxForge (Google Code archive)](https://code.google.com/archive/p/voxforge); [VoxForge about](https://voxforge.org/). Rejected (GPL, not CC0/PD).
- Wikimedia Commons Spoken Wikipedia: example files are CC BY-SA 3.0 (Danish) and CC BY-SA 4.0 (French) — [File:Spoken Wikipedia-Da-Wien.oga](https://commons.wikipedia.com/wiki/File:Spoken_Wikipedia-Da-Wien.oga), [File:FR Canadair CL-215.mp3](https://c.enwp.org/wiki/File:FR_Canadair_CL-215.mp3). Rejected as a general source. A Commons pronunciation set (>4000 clips) was reported as a relicensing request to CC0, not confirmed — [Mozilla forum](https://discourse.mozilla.org/t/adding-a-cc0-1-0-licensed-voice-database-for-odia/63857) (unverified).
- Internet Archive: LibriVox recordings are public domain in the US; other IA audio collections (Gutenberg Audio Books, Maria Lectrix, Audio Poetry, Naropa Poetics) exist, but no per-collection license or speaker count was verified — search-result secondary descriptions only.
- Project Gutenberg audio: some is machine-read (synthetic), per a secondary review. Not a human multi-speaker source.

### Inferences
- Common Voice (CC0) is the strongest candidate for many distinct speakers. Its per-speaker clip length is typically short (a few seconds per utterance), so 5 s+ continuous speech per speaker is unlikely from single sentences. This is an inference from the sentence-based format, not a verified figure.
- CoVoST adds no new audio. It reuses Common Voice clips, so it is not an independent source.
- The LibriVox pool is the main long-form public-domain source; new long-form CC0 sources were not found in this pass.

### Gaps
- Official Common Voice download route (commonvoice.mozilla.org/en/datasets and possibly Mozilla DataCollective) was NOT fetched successfully (page returned only a title). Whether it requires login is unverified. Per-language speaker counts and clip-length statistics are unverified.
- VCTK license on the Hugging Face card is unverified. The original VCTK (Edinburgh DataShare) license is not checked here and should be confirmed separately (it is commonly reported as CC BY 4.0, so probably rejected).
- Europeana public-domain audio, Zenodo CC0 search, and the HF `license:cc0-1.0` filter were not directly searched beyond one query each. Gap.
- Internet Archive public-domain filter (licenseurl) search not executed. Gap.
- No duration-per-speaker figures were verified for any source.

## Key Question 2: Speaker count, hours, clip length, language, speaker distinguishability

### Takeaway
Only CoVoST/Common Voice (CC0) has a confirmed multi-speaker count (78K speakers, 2,880 h across CoVoST). Clip-length and per-speaker minutes remain unverified; the other large multi-speaker corpora are CC-BY or CC-BY-SA.

### Cited Findings
- CoVoST: 78K speakers, 2,880 h, many languages (translation pairs into and from English) — [Meta AI CoVoST](https://ai.meta.com/tools/covost/)
- Hi-Fi TTS: 10 speakers, 291.6 h, English — [OpenSLR 109](https://openslr.org/109) (CC BY, rejected)
- LibriTTS: about 585 h English, multi-speaker — [OpenSLR 60](https://www.openslr.org/60) (CC BY, rejected)
- VCTK (HF): 110 speakers, about 44 h, English, ~400 sentences/speaker — [CSTR-Edinburgh/vctk](https://huggingface.co/datasets/CSTR-Edinburgh/vctk) (license unverified)

### Inferences
- Speaker distinguishability is high for CV/CoVoST (crowd contributors) and VCTK (110 distinct speakers), but these are sentence-level, not long continuous speech.

### Gaps
- Average and minimum clip length per source: not verified for any source.
- Per-speaker total minutes: not verified.

## Recommendation for next steps
1. Fetch the official Common Voice dataset page via a different URL or the Mozilla DataCollective page; confirm CC0 and whether login is required; fetch a small sample only.
2. Check the VCTK license on its Edinburgh DataShare page.
3. Run Zenodo and HF searches with the CC0 license filter, and an Internet Archive `licenseurl` search for public-domain speech.
4. Only if new long-form public-domain sources emerge, check clip length against the 5 s continuous-speech requirement.

No files were downloaded. Scratch folder: /tmp/cc0research (empty).
