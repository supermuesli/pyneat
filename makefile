build:
	python3 setup.py build_ext --inplace

firstBuild:
	pip3 install -r requirements.txt
	make build

clean:
	rm pyneat.c*
	rm -rf build/
	rm -rf docs/*

rebuild:
	make clean
	make build