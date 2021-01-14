This project is forked initially from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

- Test module was changed to fit with deploying Flask server.

- Checkpoint module is the place that stores the model file (.pth).

- Dataset is responded to save request files temporarily.

- Server includes 2 files: app.py and VideoProcess.py.

To run this Server, paste the command `python app.py` and enter. Run `pip install requirements.txt` first if you have not installed those packages yet.
