build:
	docker build -t ml_image .

run:
	docker run -v $$(pwd)/Project:/tmp/myapp -p 8888:8888 --rm --name ml_container ml_image

bash:
	docker exec -it ml_container /bin/bash
