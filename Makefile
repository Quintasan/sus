clusterization: clean
	docker run -ti --rm -v $(PWD):/home/docker -w /home/docker -u docker sus:latest Rscript clusterization/analyze.r

knn: clean
	python3 knn.py

clean:
	-rm -rf data_loader.pyc tmp/* tmp/.directory
