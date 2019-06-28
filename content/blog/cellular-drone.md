+++
author = "Guocheng Wei"
categories = ["Project Report"]
tags = ["Cellular Network", "MobileInsight", "LTE", "AT&T", "Latency", "Drone", "DJI", "Aerial Vehicle", "Android", "GraphQL", "Apollo", "Cloud Service", "GCP"]
date = "2019-05-31"
description = "Connect Drones Into Cellular Network"
featured = "release_drone.jpg"
featuredalt = "Release Drone"
featuredpath = "date"
linktitle = ""
Title = "Cellular Drone"
type = "post"
+++

## Motivation
Over the past few years, drones have become central to the functions of various businesses and governmental organizations and have managed to pierce through areas where certain industries were either stagnant or lagging behind.

However, almost all the commercial drones are controlled with the remote controller via the WiFi connection, which highly restricts the scope of the mobility and scale of discovery. In addition, the WiFi network is not secure. The attacker can easily disconnects the controller by changing the SSID and immediately reconnects and is then able to transmit commands though the malicious drone to the hijacked drone.

Besides, there are no centralized platform like other IoT devices to monitor and control multiple devices at the same time. All commercial drones are controlled in VLOS, Visual Line-of-Sight. In the future, drones need to have the ability to free-up pilots and go Beyond VLOS. A centralized platform can achieve this easily no matter the drone is autonomous or non-autonomous.

Also, the industry demands of the drone will not be limited to entertainment and photography. Live streaming video, package delivery and rescue mission will push the drones to be connected to the cellular network in order to free up their mobility.

## Problem
If aerial vehicles are connected to the cellular network, the first application will be the Beyond-VLOS control. BVLOS needs more sensors and cameras equipped on the drone. The reason is that the first-view camera is not enough to monitor the status of the drone. More sensors and cameras can enhance the accuracy and improve the detection of their surroundings. But it will introduced more data in the network. At the same time, the increase of amount of data should not lag the control speed.

Second concern is the collision avoidance system. BVLOS drones should have the ability to communicate with nearby drones to avoid collision. If the latency is high and drones cannot react in short period of time, accidents may happen.

The last concern is that in the future 5G network, the benchmark for the latency cellular drones is 10ms. It is hard to achieve in current LTE network.

So this project will not only focus on the development of the cellular drone application, but also analyze the end-to-end latency in order to achieve the low-latency requirement. The breakdown analysis can help us to observe the bottleneck and reveal the weakest section in this project.

## Solution
I break down the entire development into three main parts.

* Android App
	* Drone Receptor
	* Remote Controller
* Server
	* Host Database (i.e. Drone Status, User, Flight Control Messages, etc.)
	* Handle requests
* Website
	* Show data
	* Quick mutation

Android app serves two main purpose: one is the Drone Receptor (DR) that is directly connected to the drone with a micro-usb, listens to any control message sent to the drone, and sends updates of the drone status to the server; the other is the Remote Controller (RC) that is on the hand of the pilot, sends flight control messages to the server, and listens to any update from the drone.

## Challenge

Based on the experiments I did before and researches I read, higher altitude and faster speed will introduce higher latency and higher rate of packet loss. Furthermore, due to the interference of neighbor base stations, the rate of attach request failure will increase, which indicates that the handoff events take longer to complete.

So I decided to run experiments at the maximum 10m in height to decrease the influence of these issues and to make sure the flight failure cannot create any accident.

## Architecture

<img src="/img/2019/05/cloud_diagram.png" style="width: 100%;" alt="Figure 1: Architecture Diagram">
<h5 aligh="center">Figure 1: Architecture Diagram</h5>

### Specs
1. DJI Phantom 4 Pro v2.0

2. DJI Android SDK

3. GCP

	> GCP hosts my server in the cloud. I use Cloud DNS to lower latency and increase scalability, Cloud Load Balancing to scale the application on Google Compute Engine from zero to full-throttle with no pre- warming needed, and a instance group to host a kubernetes container connecting to a PostgreSQL Cloud SQL server.

4. PostgreSQL

5. GraphQL

	> The reason why I chose GraphQL api instead of REST api is because of latency. REST api uses multiple endpoints, and each endpoint will introduce latency. GraphQL api solves this problem by combining and integrating multiple endpoints into one. So it can decrease the end-to-end latency.

6. Apollo

	> Apollo is the framework of GraphQL that can help me to develop the Android app easier.

7. Android

8. AT&T

	> I run all experiments in the AT&T network.

[Cellular Drone Project Development](/blog/cellular-drone-development) is the blog focused on the details of the development. If you are interested in how I implemented all these, go check it out.

### Network Communication

As shown in the Figure 1, RC and DR communicate with the server with both HTTP and Websocket protocols. One situation is when the pilot user is controlling on the RC, flight control messagess will be sent to the server through the HTTP protocol. When the server receive any message, it will publish the message through the Websocket protocol to the subscribing DR. The other situation is when the status of the drone is updated, DR will send the updates through the HTTP protocol. When the serve receive any update, it will publish the updated status to the listening RC.

## Latency Analysis

### Overall Latency

<img src="/img/2019/05/all_boxplot_flight_ctrl_msgs.png" style="height: 30em; float: left;" alt="Figure 2: Overall Latency">
<h5 aligh="center">Figure 2: Overall Latency</h5>

This project is mainly focused on the end-to-end latency of the flight control message. So the difference between the time when a flight control message is generated on the RC and the time when the flight control message is received by the DR and executed on the drone. Therefore, there are three main factors of the latency: overhead on both phones, network latency, and overhead on the server.

I run the experiment for 10 rounds. However, 7 of them have valid and complete data. The boxplot in Figure 2 shows the median is around 500ms and the interquartile range is from 150ms to 650ms. That number is not adequate to the requirement.

In Figure 3, we can see that the latency can decrease down to 113ms and increase to 950ms. So we need to breakdown each latency to expose the bottleneck.

<img src="/img/2019/05/boxplot_every_round_flight_ctrl_msgs.png" style="width: 100%;" alt="Figure 3: Every Round End-to-End Latency">
<h5 aligh="center">Figure 3: Every Round End-to-End Latency</h5>

### Breakdown
The latency will be breakdown based on the three main factors mentioned before. After the breakdown, we will visualize which section is the root cause of the high latency.

<img src="/img/2019/05/boxplot_breakdown.png" style="width: 100%;" alt="Figure 4: Latency Breakdown">
<h5 aligh="center">Figure 4: Latency Breakdown</h5>

Figure 4 is the breakdown of one experiment. It clearly shows that the server overhead is the obstacle of the overall latency. The RC overhead, HTTP RRT and WS latency are all very stable and low. The DR overhead is stable most of the time, below 10ms, but can be over 300ms in some edge cases due to the execution time of the drone.

## Future Work

### Drawbacks
There are some main drawbacks in my architecture. The first is the two-hops communication between the RC and DR. If we want to reduce the latency, less hops should be the main thing to focus on. The second is the overhead on the server. Because it upholds more than 50% of the latency. In the future, I need to investigate the causes of the overhead and fix them. The third is the network latency. Unfortunately, Google is not a network provider. Therefore, reducing the RRT is not easy to achieve.

### Development
Live streaming video is another essential feature in the commercial drone industry. The DR is able to upload streaming video packets to the server and the server will publish the video packets to the RCs. Therefore, latency, uplink performance and downlink performance will be the KPIs.

Besides, collision avoidance system is also important. It can be divided into two parts: collision avoidance with nearby drones and with static objects. For the first part, the server is responsible to find groups of drones that are near with each other and send them immediate flight control messages to avoid the possible accident. At the same time, send warning notification messages to RCs that are in control of the drones and switch the mode to autonomous. The second part could be achieved by training convolutional neural networks of visual recognition and implementing it on the RCs so that the drones can react to the objects immediately with low latency.

# Thanks!

Photo by [Jacob Owens](https://unsplash.com/@jakobowens1?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/drone-dji?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)
