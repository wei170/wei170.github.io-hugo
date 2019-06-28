+++
author = "Guocheng Wei"
categories = ["Project Overview"]
tags = ["Demo", "Project Breakdown"]
date = "2019-06-24"
description = "Demo and break down each development of the project"
featured = "cellular_drone_dev_thumnail_0.png"
featuredalt = "cellular drone development thumbnail"
featuredpath = "date"
linktitle = ""
title = "Overview of The Cellular Drone Development"
type = "post"

+++

## Overview
I named this project *DCenter* for Drone Center, hoping that it can develop into a centralized platform that support commercial drones to fly half-autonomously with collision avoidance and stream data and 4K video to any user.

Currently, I setup a simplified infrastructure on the Heroku and GCP, and build an Android app as both the adaptor and the controller of the drone. [Cellular Drone](/blog/cellular-drone) blog will give you a more comprehensive view of the whole project and the evaluation of the current stage.

## Demo
The following is a quick demo of the project.
There are two Android phones. The right one works as the Drone Receptro (DR), which is connected to the drone via the micro-usb. The left one serves as the Remote Controller (RC), which send all control messages to the cloud server.

In order to simplify the shooting, I used DJI Assistant Simulator to visualize the movement of the drone.

At the end of the video, I will show you the website and the end-to-end latency of each control message.

{{< youtube 0nYhwhcEXK8 >}}

## Indexes
I break down the development into 5 parts:

* [DCenter Dev Part Ⅰ: GraphQL API](/blog/cellular-drone-development-1)
* [DCenter Dev Part Ⅱ: Server and Deployment](/blog/cellular-drone-development-2)
* [DCenter Dev Part Ⅲ: Drone Receptor](/blog/cellular-drone-development-3)
* [DCenter Dev Part Ⅳ: Remote Controller](/blog/cellular-drone-development-4)
* [DCenter Dev Part Ⅴ: Network Analysis](/blog/cellular-drone-development-5)
