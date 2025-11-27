FROM python:3.11-slim

# set working dir
WORKDIR /app

# copy requirements first for caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the app folder
COPY ./app ./app
COPY ./app/model ./app/model

# expose port
EXPOSE 8080

# run uvicorn
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]