# SpeechAssignment3

## Instructions to run demo
- `pydantic==1.10.7`, `gradio==3.34` are required to run the demo.
- Modify Line 24 of `model.py` to point to the appropriate Wav2Vec2 XLSR pre-trained model.
- Modify Line 10 of `gradio_demo.py` to point to the checkpoint of the FoR-finetuned model. Please find the checkpoint [here](https://iitjacin-my.sharepoint.com/:f:/g/personal/sharma_86_iitj_ac_in/Em-1lLTVAdNIkD2GmvSYjIcBhYVfrEEPNmUCDk-N3ePmMg?e=cU2Grb).
- Run `python gradio_demo.py`
