build:
	docker build -t 3d_image .

run:
	docker run -v $$(pwd)/Project:/tmp/myapp -p 8888:8888 --rm --name 3d_container 3d_image

bash:
	docker exec -it 3d_container /bin/bash