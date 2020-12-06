build:
	docker build -t luna_image .

run:
	docker run -v $$(pwd)/Project:/tmp/myapp -p 8888:8888 --rm --name luna_container luna_image