PYTHON=python3


build:
	$(PYTHON) setup.py -e .

assets/technologic.wav:
	youtube-dl --quiet \
		https://www.youtube.com/watch?v=unSzQFdr8pE \
		--extract-audio \
		--audio-format 'wav' \
		--audio-quality 0 \
		--output $@
