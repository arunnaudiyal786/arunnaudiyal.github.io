---
layout: post
title: "Stock Price Prediction with Neural Networks"
author: kunal
---

Here is a short tutorial about training an LSTM network for predicting stock price of Apple. The purpose of this blog is to provide a coding base, making it easier for anyone to start training a neural network on stocks data.  

For training this network, we take the historical data for Apple's stock (source :<a href="https://finance.yahoo.com/quote/AAPL/">Yahoo</a>) ranging from its IPO till 22nd November 2017. If anyone follows the stock market, they would know that the stock prices tend to be driven more by human sentiment in the short run. Hence they exhibit certain short term patterns, more commonly known as "Technical Indicators". Some of the most widely used technical indicators are MACD, RSI,<a href="https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/list-of-technical-indicators/">etc</a>. 

The goal of this tutorial is to capture some weekly/monthly trends in Apple's stock price over the years. Here are the broad steps you need to follow once you clone the <a href="https://github.com/ksaluja15/Stock-prediction-lstm">code repository</a>. 

* We begin by downloading Apple's stock data, which is broken up into weekly opening and closing prices.![Data]({{ "/images/AAPL.png"}})

* Once we have the data, we need to convert the data in the right format. This is done in the data_generator.py file.

* The network architecture is decribed in the model.py file.

* Simply run the train.py file to begin training.

* For inference and plotting the graphs, use the inference.py

A sample model has been provided which takes as input the data for past 4 weeks and gives the prediction for the next week. Here is the training plot:


![Data1]({{ "/images/stock-prediction/tensorboard_fig_1.png"}})

Complete code is provided here : <a href="https://github.com/ksaluja15/Stock-prediction-lstm">Stock-prediction-lstm</a>.
