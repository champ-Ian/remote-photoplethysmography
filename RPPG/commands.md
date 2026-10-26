Dowload Requirments for rPPG-TalkingFace-Detector
pip install -r rPPG-TalkingFace-Detector/requirements.txt

Download Requirments for rhythmformer
pip install -r rhythmformer/requirements.txt

Extract RPPG Waveforms
python rPPG-TalkingFace-Detector/src/extract_waveforms.py \
  --data-root Celeb-DF-v3/data \
  --repo-dir rhythmformer

Build the subject-independent split
python rPPG-TalkingFace-Detector/src/build_split.py \
  --data-root Celeb-DF-v3/data 

Run main experiments
python rPPG-TalkingFace-Detector/src/run_experiments.py \
  --data-root Celeb-DF-v3/data 

Per-method isolated training
python rPPG-TalkingFace-Detector/src/per_method_isolated.py \
  --data-root Celeb-DF-v3/data 

Generate Figures
mkdir -p ~/rPPG_Detection
ln -s /Users/ianba/Desktop/Python/RPPG/Celeb-DF-v3/data ~/rPPG_Detection/data
python rPPG-TalkingFace-Detector/src/generate_roc_curves.py 
python rPPG-TalkingFace-Detector/src/generate_waveform_comparison.py 