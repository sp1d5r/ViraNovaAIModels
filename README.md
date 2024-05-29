# ViraNova Evaluation AI

We have a dataset of videos and scores the dataset follows this schema


| id                | video_id                                | virality_score_%    | engagement_score_% | transcript                      |
|-------------------|-----------------------------------------|---------------------|--------------------|---------------------------------|
| The id of the row | video_id of the video stored on YouTube | The virality score  | Engagement score   | The transcript words themselves |


We have a few different columns the key metrics are the virality_score_% and the engagement_score_% which are calculated 
as a percentage value. 

### Engagement Score %
This is the amount of engagement the video receives normalised by the number of views it has. 

`
Engagement Score = (Total Likes + Total Comments) / Views
`

Additionally, we apply some scaling to normalise these values between 0 and 100. The end result is the following 
distribution.


### Virality Score % 

This is proposrtional to how well the video does on it's first day (how much of a push it had initially), and then 
how the video has performed over time. 

`
Virality Score = (Projected Video Views on Day 0 / Subscribers on Day 0) + (Total Video Views / Subscribers in total)
`

We also normalise this to be between 0-100. The distribution itself is a bit strange.


# Modality Types

**Textual Only - Autoencoding**: This is using auto-encoding models like BERT/RoBERTa for performing linear regression
on the dataset.

**Textual Only - Autoregressive**: This is using generative models like GPT to perform linear regression. Since it's 
stocastic we'll need to perform this several times and take averages of the outputs.

**Visual Only**: We'll be using the visual feature embeddings to extract out frame level information to try and perform 
linear regression using only visual information available.

**(Maybe) Audio Only**: using only the audio features to perform linear regression 

**MultiModal - Visual + Textual**: Comparing the video information as well as the textual information to try and 
perform linear regression on the engagement and virality metric.


# Textual Model - RoBERTa 

`/RoBERTa (Textual Autoencoding).ipynb`

Textual models follow a similiar structure, we have an encoding model which we finetune on the dataset. This model 
we'll get encodings for each word in relation to the other word. It'll produce an attention mask which we will 
be able to use to produce outputs. 

We attach a regression head to the output of the encoding stages where we try to optimise for the Mean squared error 
of the output predicted by the encodings and our regression values. 

## Graphs:

### Training and Validation (Steps)
![Training Steps](https://github.com/sp1d5r/ViraNovaAIModels/blob/0fedce40685e896e02fa1a50f81d42fcbcb2a774/graphs/RoBERTa%20/Training%20Steps.png)
![Validation Steps](https://github.com/sp1d5r/ViraNovaAIModels/blob/0fedce40685e896e02fa1a50f81d42fcbcb2a774/graphs/RoBERTa%20/Validation%20Steps.png)

### Training and Validation (Epochs)
![Training Epochs](https://github.com/sp1d5r/ViraNovaAIModels/blob/0fedce40685e896e02fa1a50f81d42fcbcb2a774/graphs/RoBERTa%20/Training%20Epochs.png)
![Validation Epochs](https://github.com/sp1d5r/ViraNovaAIModels/blob/0fedce40685e896e02fa1a50f81d42fcbcb2a774/graphs/RoBERTa%20/Validation%20Epochs.png)

### Hyperparameter Sweep
![graphs/RoBERTa /Hyperparam](https://github.com/sp1d5r/ViraNovaAIModels/blob/0fedce40685e896e02fa1a50f81d42fcbcb2a774/graphs/RoBERTa%20/Hyperparam%20Sweep.png)

# Textual Models - GPT.3.5 (BASE)

`/GPT (Textual Autoregressive).ipynb`

For this textual model we'll look at the output of GPT's auto-complete feature with specific prompts to evaluate 
how well the model predicts the output of the model. 

We'll perform this several times and then calculate the average of the outputs for this model. 




# Textual Models - 3.5 Finetuned

`/GPT (Textual Autoregressive).ipynb`

For this we'll use a finetuned model of GPT to see if that improves the performance of the baseline and by how much 
it improves the performance of the baseline. 

Similar to above we'll do this several times and evaluate the success of the average output. 


# Visual Model - ResNet 50 

`/ResNet (Visual Analysis).ipynb`

For this we'll use the pre-trained resnet model from pytorch to extract feature embeddings at a frame level and
attatch a regression head to the frames. 

We'll evaluate how well the training loss, validation loss and finally held-out test set loss performs with this.


# Visual Models - DinoV2 

`/DinoV2 (Visual Analysis).ipynb`

The most advanced base architecture for visual embeddings. and apply linear regression to this. 


# Multi-Modal Models 

(Using textual embeddings and combining these with the frame embeddings produced by the visual models.)
