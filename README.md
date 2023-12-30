# Bank-Marketing
Final Project
The information deals with phone-based marketing campaigns carried out by a Portuguese bank. In this 
project, I'm going through the entire dataset to explore the link between data from a bank's phone 
marketing efforts and the subscription status of clients. But before that, there are several missing values 
in some categorical attributes, all coded with the "unknown" label, so, I will deal with these missing 
values. I'll also create graphs to show possible connections between features like job type and deposit 
amounts, or the number of contacts made during the campaign and clients' ages.
As part of the project, I'll be presenting a classification model. This model aims to predict whether a client 
will sign up for a term deposit. Below, you'll find some details about the features of the data.
 1 - age (numeric)
 2 - job: type of job (categorical: "admin.", "unknown", "unemployed", "management", "housemaid",
"entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services") 
 3 - marital: marital status (categorical: "married", "divorced", "single"; note: "divorced" means divorced 
or widowed)
 4 - education (categorical: "unknown","secondary","primary","tertiary")
 5 - default: has credit in default? (binary: "yes","no")
 6 - balance: average yearly balance, in euros (numeric) 
 7 - housing: has housing loan? (binary: "yes","no")
 8 - loan: has personal loan? (binary: "yes","no")
 # related with the last contact of the current campaign:
 9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
 10 - day: last contact day of the month (numeric)
 11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
 12 - duration: last contact duration, in seconds (numeric)
 # other attributes:
 13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes 
last contact)
 14 - pdays: number of days that passed by after the client was last contacted from a previous campaign 
(numeric, -1 means client was not previously contacted)
 15 - previous: number of contacts performed before this campaign and for this client (numeric)
 16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown", "other", "failure",
"success")
 17 - y - has the client subscribed a term deposit? (binary: "yes","no")
