build:
	docker build -t dt_xero_image .

run:
	docker run -v $$(pwd):/tmp/myapp -p 8888:8888 --rm --name dt_xero_container dt_xero_image

bash:
	docker exec -it dt_xero_container /bin/bash
