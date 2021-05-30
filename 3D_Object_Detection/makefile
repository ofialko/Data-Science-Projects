build:
	docker build -t 3d_image .

run:
	docker run -v $$(pwd)/Project:/tmp/myapp -p 8888:8888 --rm --name 3d_container 3d_image

bash:
	docker exec -it 3d_container /bin/bash

update_container_requirements:
	docker cp 3d_container:/tmp/requirements.txt . 

init_collab:
	cp Project/Notebooks/$(notebook).ipynb ~/Google\ Drive/Colab\ Notebooks/.
	rm -rf ~/Google\ Drive/data/data_3D
	cp -r Project/data ~/Google\ Drive/data/data_3D
	rm -rf ~/Google\ Drive/src/src_3D
	cp -r Project/src ~/Google\ Drive/src/src_3D
	cp requirements_collab.txt ~/Google\ Drive/src/src_3D/.

update_drive_src:
	rm -rf ~/Google\ Drive/src/src_3D
	cp -r Project/src ~/Google\ Drive/src/src_3D
	cp requirements_collab.txt ~/Google\ Drive/src/src_3D/.

update_notebook_from_colab:
	cp ~/Google\ Drive/Colab\ Notebooks/$(notebook).ipynb Project/Notebooks/.