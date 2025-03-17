# Crypto Sentiment Analysis Project

This is my final project for MSDS 434 at Northwestern. I built a web service that checks if people are saying positive or negative things about crypto on social media. It also tries to predict if the sentiment will be more positive or negative in the future.
![image](https://github.com/user-attachments/assets/9224d06a-fa03-4a8c-ba4b-d3c91aae1468)

## What I Built

I learned a lot about cloud and CI/CD deployment in this project. Here's what I managed to get working:

- A web API using FastAPI 
- A machine learning model that can tell if a message is positive or negative
- A forecasting thing that tries to predict future sentiment
- Everything runs in Docker containers
- It's all running on AWS EC2

## How to Run It
http://ec2-35-86-159-254.us-west-2.compute.amazonaws.com:8080/predict\
http://ec2-35-86-159-254.us-west-2.compute.amazonaws.com:8080/forecast_chart/7\

If you want to try this out locally:

1. Clone this repo
2. Make sure you have Docker installed
3. Run `docker-compose up --build`

## Project Structure

Here's how I organized my files:

app/
 --models/              # Where the ML models live\
-- app.py              # Main API code\
-- train_forecast_model.py    # Creates the forecast model\
-- train_sentiment_model.py   # Creates the sentiment model\
-- requirements.txt    # Python packages I used\

## What the API Does

You can:
- Send a message to `/predict` to check its sentiment
- Use `/forecast` to see predicted sentiment for the next few days
- Check out `/forecast_chart/{days}` to see a cool graph of the predictions

The sentiment predictions aren't perfect (I only had data from January 2021 to work with), but it was fun learning how to build this.

## AWS Deployment

I'm using:
- An EC2 instance to run everything
- S3 to store some data
- GitHub Actions for automatic deployment

Getting this deployed was tricky. I had to learn about:
- Docker and docker-compose
- How to set up an EC2 instance
- How to use GitHub Actions 
- Managing AWS credentials safely

## What I Learned

This project taught me a lot about:
1. Building and deploying web services
2. Working with AWS
3. Using Docker
4. Setting up automated deployments
5. How hard it is to get good training data!

## Future Ideas

If I had more time, I'd like to:
1. Get real-time data from Twitter or Reddit
2. Make the predictions more accurate
3. Add more cryptocurrencies
4. Make the charts look nicer

## Known Issues

- The forecast model only has data from January 2021
- Sometimes the Docker build fails and needs to be restarted
- The GitHub Actions deployment can be flaky

## Course Info
MSDS 434 - Data Science & Cloud Computing
Northwestern University
Winter 2025
