#!/bin/bash
npm run build
sudo docker build -t konglsh96/movie_rec_bert:front . 
sudo docker push konglsh96/movie_rec_bert:front
