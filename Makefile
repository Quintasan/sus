clusterization:
	rm -f tmp/*
	docker run -ti --rm -v $(PWD):/home/docker -w /home/docker -u docker sus:latest Rscript clusterization/analyze.r
