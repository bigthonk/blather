update:
		-rm -r dist
		-rm -r blather.egg-info
		python setup.py sdist
		python -m twine upload dist/*
