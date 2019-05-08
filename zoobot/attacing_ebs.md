AWS EBS is not automatically saving your root directory. 

It's important to mount EBS before doing anything else, especially on a Spot instance.


## Setup

`ebs_drive=/dev/xvdb`

`ebs_loc=ebs`

## From Blank Snapshot

I have made a blank snapshot to use in future called `blank_30gb`, with id `snap-05017edd4f14645f2`.

`sudo mount $ebs_drive $ebs_loc`


## Create new blank EBS

*This should not be necessary. I have made a blank snapshot to use.*

`cd && sudo mkfs -t ext4 $ebs_drive && sudo mkdir $ebs_loc && sudo mount $ebs_drive $ebs_loc && sudo chmod a+w $ebs_loc`

This can be used to [reset the TensorFlow environment](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-tensorflow.html) if I uninstalled the GPU version. Oops.

The AWS guide is [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html).

Follow it carefully. 

Step 8 is important to make the new filesystem writeable. Use 

`sudo chmod a+w ebs`
To set a(ll) users able to w(rite). Based on [this](https://www.linode.com/docs/tools-reference/tools/modify-file-permissions-with-chmod/) explanation of chmod.

Afterwards, consider making a snapshot. 