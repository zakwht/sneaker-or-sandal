.phony all: run
run:
	FLASK_APP=app.py FLASK_ENV=development python3 -m flask run

deploy:
	git push heroku main
	heroku open