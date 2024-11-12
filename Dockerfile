
FROM python:3.9-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


WORKDIR /app


COPY requirements.txt /app/requirements.txt
COPY major_d24csa006.py /app/major_d24csa006.py


RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 5000

CMD ["python", "major_d24csa006.py"]
