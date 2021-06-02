project := $(shell basename $(abspath $(dir $$PWD)))

src_colab = src_$(project)
data_colab = data_$(project)
notebook_colab = notebook_$(project)
image = image_$(project)
container = container_$(project)

gdrive_path = Google\ Drive\ \(oleksandrfialko@gmail.com\)

env:
	echo $(gdrive_path)

build:
	docker build -t $(image) .

run:
	docker run -v $$(pwd)/Project:/tmp/myapp -p 8888:8888 --rm --name $(container) $(image)

bash:
	docker exec -it $(container) /bin/bash

update_container_requirements:
	docker cp $(container):/tmp/requirements.txt . 

init_collab:
	cp Project/Notebooks/$(notebook_colab).ipynb ~/$(gdrive_path)/Colab\ Notebooks/.
	mkdir -p ~/$(gdrive_path)/data/$(data_colab)
	mkdir -p ~/$(gdrive_path)/src/$(src_colab)
	rm -rf ~/$(gdrive_path)/data/$(data_colab)
	cp -r Project/data ~/$(gdrive_path)/data/$(data_colab)
	rm -rf ~/$(gdrive_path)/src/$(src_colab)
	cp -r Project/src ~/$(gdrive_path)/src/$(src_colab)
	cp requirements_collab.txt ~/$(gdrive_path)/src/$(src_colab)/.

update_drive_src:
	rm -rf ~/$(gdrive_path)/src/$(src_colab)
	cp -r Project/src ~/$(gdrive_path)/src/$(src_colab)
	cp requirements_collab.txt ~/$(gdrive_path)/src/$(src_colab)/.

update_notebook_from_colab:
	cp ~/$(gdrive_path)/Colab\ Notebooks/$(notebook_colab).ipynb Project/Notebooks/.