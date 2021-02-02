library(BuenaVista)
library(tidyverse)

# load in data
set.seed(seed = 1234)
getwd()
setwd(dir = "C:/Users/oisin_000/Documents/R/Kaggle Competitions/Titanic Competition/")
train <- read.csv(file = "train.csv", header = TRUE)
test <- read.csv(file = "test.csv", header = TRUE)
# Combine Training and Test Sets
Survived <- rep(x = NA, times = 418)
test <- cbind(Survived, test)
data <- rbind(train, test)
# some initial data processing
str(data)
data$Name <- as.character(data$Name)
data$Cabin <- as.character(data$Cabin)
data$Ticket <- as.character(data$Ticket)
data$Survived <- as.factor(data$Survived)
data$Pclass <- as.factor(data$Pclass)
data$SibSp <- as.integer(data$SibSp)
data$Parch <- as.integer(data$Parch)

###############################
## Explorative Data Analysis ##
###############################

#-- derive descriptive statistics
descriptive_statistics(dataset = data, 
                       type = "numeric")
descriptive_statistics(dataset = data, 
                       type = "factor")

#-- construct visualisations
# Single variables
visualise_variables_x(dataset = data[sample(1:nrow(data), 
                                            size = 1000, 
                                            replace = FALSE),])
# Pairs of varibales
visualise_variables_xx(dataset = data[sample(1:nrow(data), 
                                             size = 1000, 
                                             replace = FALSE),])
# vs response
visualise_variables_xx(dataset = data[sample(1:nrow(data), size = 1000, 
                                             replace = FALSE),],
                       y_index = 2)

#-- correlations
tests_cors(dataset = data)

#-- associations
tests_chisq(dataset = data,
            simulate.p.value = TRUE)
tests_chisq(dataset = data,
            simulate.p.value = TRUE,
            y_index = 2)

########################
## Feature Processing ##
########################

# impute the missing values for embarked
descriptive_statistics(data, type = "factor")
str(data)
data <- imputate_variables(dataset = data, y_index = 12, type = "mode")

# impute the missing values for fare
descriptive_statistics(data, type = "numeric")
data <- imputate_variables(dataset = data, y_index = 10, type = "mean")

# impute the missing values for age
descriptive_statistics(data, type = "numeric")
data <- imputate_variables(dataset = data[,-c(1,2,10)], y_index = 10, type = "mice", method = "norm.predict")

##########
## Name ##
##########

library(stringi)

# method 1
# Create a character vector of name
Name <- as.character(data$Name)
# Search each for an appropriate title and replace name with said title
Name[which(stri_detect_fixed(str = Name, pattern = " Capt. "))] <- "Capt"
Name[which(stri_detect_fixed(str = Name, pattern = " Col. "))] <- "Col"
Name[which(stri_detect_fixed(str = Name, pattern = " Major. "))] <- "Major"
Name[which(stri_detect_fixed(str = Name, pattern = " Dr. "))] <- "Dr"
Name[which(stri_detect_fixed(str = Name, pattern = " Rev. "))] <- "Rev"
Name[which(stri_detect_fixed(str = Name, pattern = " Jonkheer. "))] <- "Jonhkheer"
Name[which(stri_detect_fixed(str = Name, pattern = " Sir. "))] <- "Sir"
Name[which(stri_detect_fixed(str = Name, pattern = " Don. "))] <- "Don"
Name[which(stri_detect_fixed(str = Name, pattern = " Countess. "))] <- "Countess"
Name[which(stri_detect_fixed(str = Name, pattern = " Dona. "))] <- "Dona"
Name[which(stri_detect_fixed(str = Name, pattern = " Master. "))] <- "Master"
Name[which(stri_detect_fixed(str = Name, pattern = " Lady. "))] <- "Lady"
Name[which(stri_detect_fixed(str = Name, pattern = " Mrs. "))] <- "Mrs"
Name[which(stri_detect_fixed(str = Name, pattern = " Mme. "))] <- "Mrs"
Name[which(stri_detect_fixed(str = Name, pattern = " Mlle. "))] <- "Miss"
Name[which(stri_detect_fixed(str = Name, pattern = " Ms. "))] <- "Miss"
Name[which(stri_detect_fixed(str = Name, pattern = " Miss. "))] <- "Miss"
Name[which(stri_detect_fixed(str = Name, pattern = " Mr. "))] <- "Mr"
# update the data
Name <- as.factor(Name)
data$Name <- Name

# method 2
# Create a character vector of name
Name <- as.character(data$Name)
# Further Condensed Titles
# Search each for an appropriate title and replace name with said title
Name[which(stri_detect_fixed(str = Name, pattern = " Capt. "))] <- "Officer"
Name[which(stri_detect_fixed(str = Name, pattern = " Col. "))] <- "Officer"
Name[which(stri_detect_fixed(str = Name, pattern = " Major. "))] <- "Officer"
Name[which(stri_detect_fixed(str = Name, pattern = " Dr. "))] <- "Officer"
Name[which(stri_detect_fixed(str = Name, pattern = " Rev. "))] <- "Officer"
Name[which(stri_detect_fixed(str = Name, pattern = " Jonhkeer. "))] <- "Royalty"
Name[which(stri_detect_fixed(str = Name, pattern = " Sir. "))] <- "Royalty"
Name[which(stri_detect_fixed(str = Name, pattern = " Don. "))] <- "Royalty"
Name[which(stri_detect_fixed(str = Name, pattern = " Countess. "))] <- "Royalty"
Name[which(stri_detect_fixed(str = Name, pattern = " Dona. "))] <- "Royalty"
Name[which(stri_detect_fixed(str = Name, pattern = " Master. "))] <- "Royalty"
Name[which(stri_detect_fixed(str = Name, pattern = " Lady. "))] <- "Royalty"
Name[which(stri_detect_fixed(str = Name, pattern = " Mrs. "))] <- "Mrs"
Name[which(stri_detect_fixed(str = Name, pattern = " Mme. "))] <- "Mrs"
Name[which(stri_detect_fixed(str = Name, pattern = " Mlle. "))] <- "Miss"
Name[which(stri_detect_fixed(str = Name, pattern = " Ms. "))] <- "Miss"
Name[which(stri_detect_fixed(str = Name, pattern = " Miss. "))] <- "Miss"
Name[which(stri_detect_fixed(str = Name, pattern = " Mr. "))] <- "Mr"
# update the data
Name <- as.factor(Name)
data$Name <- Name



###########
## Cabin ##
###########

# NOTE: some classification models cannot handle more than 53 categories e.g. Random Forests

# notice cabin has missing levels as well
summary(Cabin)
Cabin <- data$Cabin
levels(x = Cabin)
Cabin

# option 1: add in "unknown" label
# option 2: add in "unknown" label and aggregate up to A, B, C, E, F Cabins

# option 1:
Cabin <- data$Cabin
levels(x = Cabin)
Cabin
# create a new level called "unknown"
levels(x = Cabin) <- c("Unknown", levels(x = Cabin))
levels(x = Cabin)
# add in unknown level
Cabin[which(Cabin == "")] <- "Unknown"
summary(Cabin)
# update factor levels to remove " " level
Cabin <- factor(Cabin)
levels(Cabin)
data$Cabin <- Cabin
levels(data$Cabin)
summary(Cabin)
Cabin

# option 2:
Cabin <- data$Cabin
levels(x = Cabin)
Cabin
# create a new level called "unknown"
levels(x = Cabin) <- c("Unknown", levels(x = Cabin))
levels(x = Cabin)
# add in unknown level
Cabin[which(Cabin == "")] <- "Unknown"
summary(Cabin)
# update factor levels to remove " " level
Cabin <- factor(Cabin)
levels(Cabin)
data$Cabin <- Cabin
levels(data$Cabin)
summary(Cabin)
Cabin
# aggregating up tp A, B, C, D, E
# turn Cabin into a character variable
# this facilitates stringi operations
Cabin <- as.character(Cabin)
Cabin
Cabin[which(stri_detect_fixed(str = Cabin, pattern = "A"))] <- "A"
Cabin[which(stri_detect_fixed(str = Cabin, pattern = "B"))] <- "B"
Cabin[which(stri_detect_fixed(str = Cabin, pattern = "C"))] <- "C"
Cabin[which(stri_detect_fixed(str = Cabin, pattern = "D"))] <- "D"
Cabin[which(stri_detect_fixed(str = Cabin, pattern = "E"))] <- "E"
Cabin[which(stri_detect_fixed(str = Cabin, pattern = "F"))] <- "F"
Cabin[which(stri_detect_fixed(str = Cabin, pattern = "G"))] <- "G"
Cabin
# turn cabin into a factor variable
Cabin <- as.factor(x = Cabin)
Cabin
data$Cabin <- Cabin

############
## Ticket ##
############

# Option 1: keep the prefix of each ticket
# Option 2: remove the prefix of each ticket and convert variable to a numeric variable


# option 1: remove the digits from each ticket and keep the prefix


# method 1 - keep prefix
Ticket <- as.character(data$Ticket)
# removing each prefix one by one
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "STON/O2. "))] <- "STON/O2."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SC/Paris "))] <- "SC/Paris"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "S.C./A.4. "))] <- "S.C./A.4."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SOTON/OQ "))] <- "SOTON/OQ"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "W.E.P. "))] <- "W.E.P."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "STON/O 2. "))] <- "STON/O2."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SOTON/O.Q. "))] <- "SOTON/O.Q."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SC/PARIS "))] <- "SC/PARIS"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SOTON/O2 "))] <- "SOTON/O2"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "S.C./PARIS "))] <- "S.C./PARIS"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "C.A./SOTON "))] <- "C.A./SOTON"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "STON/OQ. "))] <- "STON/OQ."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "S.O./P.P. "))] <- "S.O./P.P."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "S.O.P. "))] <- "S.O.P."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "Basle "))] <- "Basle"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A/5 "))] <- "A/5"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A/5. "))] <- "A/5."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "C.A. "))] <- "C.A."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A./5. "))] <- "A./5."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A/4. "))] <- "A/4."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "CA "))] <- "CA"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "S.O.C. "))] <- "S.O.C."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SO/C "))] <- "SO/C"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "W./C. "))] <- "W./C."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A4. "))] <- "A4."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A.5. "))] <- "A.5."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SC/AH "))] <- "SC/AH"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "F.C.C. "))] <- "F.C.C."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A/S "))] <- "A/S"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A/4 "))] <- "A/4"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "WE/P "))] <- "WE/P"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "./PP "))] <- "./PP"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "CA. "))] <- "CA."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "F.C. "))] <- "F.C."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SC/A4 "))] <- "SC/A4"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "AQ/4 "))] <- "AQ/4"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A. 2. "))] <- "A. 2."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "AQ/3. "))] <- "AQ/3."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "S.P. "))] <- "S.P."
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SCO/W "))] <- "SCO/W"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SC/A.3 "))] <- "SC/A.3"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "W/C "))] <- "W/C"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "Fa "))] <- "Fa"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "PP "))] <- "PP"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "PC "))] <- "PC"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "C "))] <- "C"
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "P "))] <- "P"
data$Ticket <- Ticket
?ifelse
# create vector of available prefixes
prefixes <- c("A/5", "PC", "STON/O2.", "PP", "A/5.", "C.A.", "A./5.", "SC/Paris", "S.C./A.4.", "A/4.", "CA", "S.O.C.", "SO/C", "W./C.", "SOTON/OQ", "W.E.P.", "STON/O2.", "A4.", "SOTON/O.Q.", "SC/PARIS", "S.O.P.", "A.5.", "Fa", "SC/AH", "F.C.C.", "A/S", "Basle", "A/4", "WE/P", "./PP", "S.O./P.P.", "CA.", "F.C.", "SOTON/O2", "S.C./PARIS", "C.A./SOTON", "STON/OQ.", "SC/A4", "AQ/4", "A. 2.", "AQ/3.", "S.P.", "SCO/W", "SC/A.3", "W/C", "C", "P")
length(prefixes)
# convert pure numeric tickets into 'no-prefix' tickets
data$Ticket <- ifelse(test = Ticket %in% prefixes, yes = Ticket, no = "no-prefix")
data$Ticket <- as.factor(data$Ticket)
summary(data$Ticket)
typeof((data$Ticket))

# method 2
# remove after prefixs past to the " " character
?gsub()
Ticket <- as.character(data$Ticket)
Ticket <- gsub(pattern = ".* ", replacement ="", x = Ticket)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "LINE"))] <- 1
data$Ticket <- Ticket
# convert ticket to a numeric variable
Ticket <- as.numeric(x = Ticket)
data$Ticket <- Ticket
anyNA(x = Ticket)
which(is.na(x = Ticket))
Ticket[which(is.na(x = Ticket))]


# option 2: remove the prefix of each ticket and convert variable to a numeric variable


# remove prefix from tickets
# could search strings with begin "..."

# method 1
Ticket <- as.character(data$Ticket)
# removing each prefix one by one
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A/5 "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A/5 "))], from = 5)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "PC "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "PC "))], from = 4)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "STON/O2. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "STON/O2. "))], from = 10)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "PP "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "PP "))], from = 4)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A/5. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A/5. "))], from = 6)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "C.A. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "C.A. "))], from = 6)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A./5. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A./5. "))], from = 7)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SC/Paris "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SC/Paris "))], from = 10)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "S.C./A.4. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "S.C./A.4. "))], from = 11)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A/4. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A/4. "))], from = 6)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "CA "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "CA "))], from = 4)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "S.O.C. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "S.O.C. "))], from = 8)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SO/C "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SO/C "))], from = 6)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "W./C. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "W./C. "))], from = 7)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SOTON/OQ "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SOTON/OQ "))], from = 10)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "W.E.P. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "W.E.P. "))], from = 8)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "STON/O 2. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "STON/O 2. "))], from = 11)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A4. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A4. "))], from = 5)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "C "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "C "))], from = 3)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SOTON/O.Q. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SOTON/O.Q. "))], from = 12)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SC/PARIS "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SC/PARIS "))], from = 10)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "S.O.P. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "S.O.P. "))], from = 8)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A.5. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A.5. "))], from = 5)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "Fa "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "Fa "))], from = 4)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SC/AH "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SC/AH "))], from = 7)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "F.C.C. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "F.C.C. "))], from = 8)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A/S "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A/S "))], from = 5)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "Basle "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "Basle "))], from = 7)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A/4 "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A/4 "))], from = 5)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "WE/P "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "WE/P "))], from = 6)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "./PP "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "./PP "))], from = 6)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "S.O./P.P. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "S.O./P.P. "))], from = 11)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "CA. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "CA. "))], from = 5)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "F.C. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "F.C. "))], from = 6)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SOTON/O2 "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SOTON/O2 "))], from = 10)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "S.C./PARIS "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "S.C./PARIS "))], from = 12)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "P "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "P "))], from = 3)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "C.A./SOTON "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "C.A./SOTON "))], from = 12)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "STON/OQ. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "STON/OQ. "))], from = 10)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SC/A4 "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SC/A4 "))], from = 7)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "AQ/4 "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "AQ/4 "))], from = 6)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A. 2. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "A. 2. "))], from = 7)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "AQ/3. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "AQ/3. "))], from = 7)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "S.P. "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "S.P. "))], from = 6)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SCO/W "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SCO/W "))], from = 7)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SC/A.3 "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "SC/A.3 "))], from = 8)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "W/C "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "W/C "))], from = 5)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "C "))] <- stri_sub(str = Ticket[which(stri_detect_fixed(str = Ticket, pattern = "C "))], from = 3)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "LINE"))] <- 1
data$Ticket <- Ticket
# convert ticket to a numeric variable
Ticket <- as.numeric(x = Ticket)
data$Ticket <- Ticket
anyNA(x = Ticket)
which(is.na(x = Ticket))
Ticket[which(is.na(x = Ticket))]

# method2
# remove prefixs up to the " " character
?gsub()
Ticket <- as.character(data$Ticket)
Ticket <- gsub(pattern = ".* ", replacement ="", x = Ticket)
Ticket[which(stri_detect_fixed(str = Ticket, pattern = "LINE"))] <- 1
data$Ticket <- Ticket
# convert ticket to a numeric variable
Ticket <- as.numeric(x = Ticket)
data$Ticket <- Ticket
anyNA(x = Ticket)
which(is.na(x = Ticket))
Ticket[which(is.na(x = Ticket))]

###########################################################################################################################
## Convert to Categorical Variables ##################################################################################
##########################################################################################################################

# Fare
# Adittional option
# we can convert Fare into a categorical variable
?cut()
max(data$Fare)
data$ Fare <- cut(x = data$Fare, breaks = seq(from = 0, to = 520, by = 10), right = F)
anyNA(x = data$Fare)

# Age
# Additional Option
# convert age into a categorical variable
?cut()
max(data$Age)
data$Age <- cut(x = data$Age, breaks = seq(from = 0, to = 85, by = 5), right = F)
anyNA(data$Age)

#################################################################################################################################
## Standardising Numeric Data (If Applicable)###################################################################################################
#################################################################################################################################

# Two Options
# Option 1: standardise data to be on same scale
# Option 2: don't standardise data

# Option 1

# standardise Age, Fare, Ticket to have mean 0 and standard deviation 1
?scale()

# Age
data$Age <- scale(data$Age)
round(mean(data$Age, na.rm = T))
var(data$Age, na.rm = T)

# Fare 
data$Fare <- scale(data$Fare)
round(mean(data$Fare, na.rm = T))
var(data$Fare, na.rm = T)

# Ticket
data$Ticket <- scale(data$Ticket)
round(mean(data$Ticket, na.rm = T))
var(data$Ticket, na.rm = T)

# Option 2

# standardising isn't necessary as Age and Fare are roughly on the same scale
# see an update of the data so far
summary(data)
str(data)

#################################################################################################################################
## Coding Dummy Variables (If Applicable)###################################################################################################
#################################################################################################################################

catdata <- data[ , -(1:2)]
library(dummy)
data <- cbind(data$PassengerId, data$Survived, dummy(x = catdata))
colnames(data)[1:2] <- c("PassengerId", "Survived")

# note that some columns only have one level "0"
# classification models such as Random Forests and Support Vector Machines require features with 2 or more levels
# check dummy levels
for(i in 3:ncol(data)) {
  print(levels(data[,i]))
}
# input new dummy labels
for(i in 3:ncol(data)) {
  levels(data[ ,i]) <- c("0", "1")
}
# check new dummy levels
for(i in 3:ncol(data)) {
  print(levels(data[,i]))
}

############################################################################################################################
## Partitioning the data ###################################################################################################
############################################################################################################################

# ALL VALUES INCLUDED
# separate the test data from the training data
# testing set
ptest <- data[((nrow(data) - 418) + 1):nrow(data),-c(1, 2)]
# training set
pdata <- data[1:(nrow(data) - 418), -1]
# We need to turn Survived into a factor
pdata$Survived <- as.factor(pdata$Survived)
levels(pdata$Survived) <- 0:1
levels(pdata$Survived)
summary(pdata$Survived)
summary(pdata)

# Partitioning training data
# Option 1: Partition 7:3
# Option 2: Partition 6:4
# Option 3: k-cross fold validation

# Option 1: Partition 7:3
# further divide the data into training and validation sets 70%-30%
ptrain <- pdata[1:(nrow(pdata) * 0.7), ]
pvalid <- pdata[(nrow(pdata) * 0.7):nrow(pdata), -1]
pvalidlabels <- pdata[((nrow(pdata) * 0.7):nrow(pdata)), 1]

# Option 2: Partition 6:4
# further divide the data into training and validation sets 60%-40%
ptrain <- pdata[1:(nrow(pdata) * 0.6), ]
pvalid <- pdata[(nrow(pdata) * 0.6):nrow(pdata), -1]
pvalidlabels <- pdata[((nrow(pdata) * 0.6):nrow(pdata)), 1]

# Option 3: K-cross fold validation
ptrain <- pdata
# are there any NA values
summary(ptrain)
anyNA(ptrain)
# K-fold Cross validation
bptrain <- ptrain

# note the length of the data
# depending on the options taking this will vary significantly

########################################################################################################################
## Sampling the Data (If Applicable) ###################################################################################################
########################################################################################################################

library(ROSE)

?ovun.sample()
# NOTE: ovun.sample orders the data
# NOTE: ovun.sample removes any missing NA values
# NOTE: sampling is not necessary for K-Cross Fold Validation

# three options
# (1) over sample the data - add no observations to balance distribtuion
# (2) under sample the data - remove yes observations to balance distribution
# (3) mixture sample the data - mixture of under and over sampling

# Option 1: Over sample the data
bptrain <- ovun.sample(Survived ~ ., data = ptrain, method = "over", na.action = "na.pass")
bptrain <- bptrain$data
# Bar Chart of Survived
ggplot(data = bptrain, mapping = aes(x = Survived, fill = Survived)) + geom_bar(fill = c("red", "green")) + labs(title = "Bar Chart of Survival", x = "Survived", y = "Count") 
# we can see that the data is now roughly balanced
summary(bptrain$Survived)

# Option 2: Under sample the data
bptrain <- ovun.sample(Survived ~ ., data = ptrain, method = "under", na.action = "na.pass")
bptrain <- bptrain$data
# Bar Chart of Survived
ggplot(data = bptrain, mapping = aes(x = Survived, fill = Survived)) + geom_bar(fill = c("red", "green")) + labs(title = "Bar Chart of Survival", x = "Survived", y = "Count") 
# we can see that the data is now roughly balanced
summary(bptrain$Survived)

#########################################################################################################################
## Randomise the Training Data (If Applicable) ###################################################################################################
#########################################################################################################################

# As noted above the sampling with ovun.sample orders the data
head(bptrain)
# currently the data is ordered due to ovun.sample() command
# as such we need to randomise the data
u <- runif(n = nrow(bptrain))
bptrain <- bptrain[order(u), ]
head(bptrain)
View(bptrain)

########################################################################################################################################
## Classification Models with rminer ###################################################################################################
########################################################################################################################################

library(rminer)

library(caret)

# fit the following models
# (1) Decision Tree
# (2) Random Forests
# (3) Naive Bayes Classifiers
# (4) K-Nearest Neighbours
# (5) Support Vector Machines
# (6) Logistic Regression
# (7) Conditional Inference Trees

View(data)

# NOTE: report the default model settings

#########################
## Decision Tree Model ##
#########################

#NOTE: decision trees can handle missing data

# fit decision tree model using fit()
?fit()
dtree <- fit(Survived ~ ., data = bptrain, model = "rpart", task = "class", method = "class", parms = list(split = "gini"))
dtree <- fit(Survived ~ ., data = bptrain, model = "rpart", task = "class", method = "class", parms = list(split = "information"))

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = dtree, newdata = pvalid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, pvalidlabels))

# measure input importance 
idtree <- Importance(M = dtree, data = bptrain)
idtree$value
# numeric vector with the computed sensitivity analysis measure for each variable
idtree$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

# k-fold cross validation
?crossvaldata()
cvdt <- crossvaldata(x = Survived ~. , data = bptrain, theta.fit = fit, theta.predict = predict, task = "class", model = "rpart", ngroup = 10, parms = list(split = "gini"))
summary(cvdt)
# Performance Metrics
?mmetric
mmetric(y = bptrain$Survived, cvdt$cv.fit, metric = "ALL")
mmetric(y = bptrain$Survived, cvdt$cv.fit, metric = "CONF")
# Graphs
mgraph(y = bptrain$Survived, x = cvdt$cv.fit, graph = "ROC", TC = 0, main = "ROC Chart of Decision Tree Model", baseline = T, Grid = 10)


#########################
## Random Forest Model ##
#########################

# NOTE: Random Forests cannot handle missing data

# fit random forests model using fit()
?fit()
rforest <- fit(Survived ~ ., data = bptrain, model = "randomForest", task = "class", ntree = 500, replace = T, strata = T)

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = rforest, newdata = pvalid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, pvalidlabels))

# measure input importance 
irforest <- Importance(M = rforest, data = bptrain)
irforest$value
# numeric vector with the computed sensitivity analysis measure for each variable
irforest$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

# k-fold cross validation
?crossvaldata()
cvrf <- crossvaldata(x = Survived ~. , data = bptrain, theta.fit = fit, theta.predict = predict, task = "class", model = "randomForest", ngroup = 10, ntree = 500, replace = T, strata = T)
summary(cvrf)
# Performance Metrics
?mmetric
mmetric(y = bptrain$Survived, cvrf$cv.fit, metric = "ALL")
mmetric(y = bptrain$Survived, cvrf$cv.fit, metric = "CONF")
# Graph
mgraph(y = bptrain$Survived, x = cvrf$cv.fit, graph = "ROC", TC = 0, main = "ROC Chart of Random Forests Model", baseline = T, Grid = 10)


###################
## Bayes Network ##
###################

# NOTE: Naive Bayes Classifiers can handle missing data

# fit Naive Bayes Classifiers model using fit()
?fit()
nbayes <- fit(Survived ~ ., data = bptrain, model = "naive", task = "class")

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = nbayes, newdata = pvalid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, pvalidlabels))

# measure input importance 
inbayes <- Importance(M = nbayes, data = bptrain)
inbayes$value
# numeric vector with the computed sensitivity analysis measure for each variable
inbayes$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

# k-fold cross validation
?crossvaldata()
cvnb <- crossvaldata(x = Survived ~. , data = bptrain, theta.fit = fit, theta.predict = predict, task = "class", model = "naive", ngroup = 10)
summary(cvnb)
# Performance Metrics
?mmetric
mmetric(y = bptrain$Survived, cvnb$cv.fit, metric = "ALL")
mmetric(y = bptrain$Survived, cvnb$cv.fit, metric = "CONF")
# Graphs
mgraph(y = bptrain$Survived, x = cvnbcv.fit, graph = "ROC", TC = 0, main = "ROC Chart of Bayesian Network Model", baseline = T, Grid = 10)


##############################
## Generalized Linear Model ##
##############################

# fit Generalized Linear Mode model using fit()
?fit()
cvglm <- fit(Survived ~ ., data = bptrain, model = "cv.glmnet", task = "class")

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = cvglm, newdata = pvalid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, pvalidlabels))

# measure input importance 
icvglm <- Importance(M = cvglm, data = bptrain)
icvglm$value
# numeric vector with the computed sensitivity analysis measure for each variable
icvglm$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

# k-fold cross validation
?crossvaldata()
cvcvglm <- crossvaldata(x = Survived ~. , data = bptrain, theta.fit = fit, theta.predict = predict, task = "class", model = "cv.glmnet", ngroup = 10)
summary(cvcvglm)
# Performance Metrics
?mmetric
mmetric(y = bptrain$Survived, cvcvglm$cv.fit, metric = "ALL")
mmetric(y = bptrain$Survived, cvcvglm$cv.fit, metric = "CONF")
# Graphs
mgraph(y = bptrain$Survived, x = cvcvglm$cv.fit, graph = "ROC", TC = 0, main = "ROC Chart of General Linear Model", baseline = T, Grid = 10)


##########################
## K-Nearest Neighbours ##
##########################

# Knn cannot handle missing data

# fit K-Narest Neighbours model using fit()
?fit()
knnm <- fit(Survived ~ ., data = bptrain, model = "knn", task = "class", k = 9,  distance = 1, kernel = "rank")

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = knnm, newdata = pvalid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, pvalidlabels))

# measure input importance 
iknnm <- Importance(M = knnm, data = bptrain)
iknnm$value
# numeric vector with the computed sensitivity analysis measure for each variable
iknnm$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

# k-fold cross validation
?crossvaldata()
cvknn <- crossvaldata(x = Survived ~. , data = bptrain, theta.fit = fit, theta.predict = predict, task = "class", model = "knn", ngroup = 10, k = 9, distance = 1, kernel = "rank")
summary(cvknn)
# Perfromance Metrics
?mmetric
mmetric(y = bptrain$Survived, cvknn$cv.fit, metric = "ALL")
mmetric(y = bptrain$Survived, cvknn$cv.fit, metric = "CONF")
# Graphs
mgraph(y = bptrain$Survived, x = cvknn$cv.fit, graph = "ROC", TC = 0, main = "ROC Chart of K-Nearest Neighbour Model", baseline = T, Grid = 10)

#############################
## Support Vector Machines ##
#############################

library(kernlab)

# Support Vector Machines cannot handle missing data

# fit support vector machines model using fit()
?fit()
svmm <- fit(Survived ~ ., data = bptrain, model = "ksvm", task = "class", kernel = "rbfdot", kpar = list(sigma = 0.08333333), C = 1)

# finding optimal sigma
sigest(x = Survived ~ . , data = bptrain)

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = svmm, newdata = pvalid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, pvalidlabels))

# measure input importance 
isvmm <- Importance(M = svmm, data = bptrain)
isvmm$value
# numeric vector with the computed sensitivity analysis measure for each variable
isvmm$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

# k-fold cross validation
?crossvaldata()
cvsvm <- crossvaldata(x = Survived ~. , data = bptrain, theta.fit = fit, theta.predict = predict, task = "class", model = "ksvm", ngroup = 10, kernel = "rbfdot", kpar = list(sigma = 0.08333333), C = 1)
cvsvm <- crossvaldata(x = Survived ~. , data = bptrain, theta.fit = fit, theta.predict = predict, task = "class", model = "ksvm", ngroup = 10, kernel = "rbfdot", kpar = list(default = "automatic"), C = 1)
summary(cvsvm)
attributes(cvsvm)
# performance measurements
?mmetric()
mmetric(y = bptrain$Survived, x = cvsvm$cv.fit, metric = "ALL")
mmetric(y = bptrain$Survived, x = cvsvm$cv.fit, metric = "CONF")
# graphs
mgraph(y = bptrain$Survived, x = cvsvm$cv.fit, graph = "ROC", TC = 0, main = "ROC Chart of Support Vector Machine Model", baseline = T, Grid = 10)

mgraph(y = bptrain$Survived, x = cvsvm$cv.fit, graph = "IMP")
mgraph(y = bptrain$Survived, x = cvsvm$cv.fit, graph = "LIFT")
?mgraph()

#########################
## Logistic Regression ##
#########################

# logistic regression can handle missing data

# fit logistic regression model using fit()
?fit()
lr <- fit(Survived ~ ., data = bptrain, model = "lr", task = "class")

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = lr, newdata = pvalid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, pvalidlabels))

# measure input importance 
ilr <- Importance(M = lr, data = bptrain)
ilr$value
# numeric vector with the computed sensitivity analysis measure for each variable
ilr$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

# k-fold cross validation
?crossvaldata()
cvlr <- crossvaldata(x = Survived ~. , data = bptrain, theta.fit = fit, theta.predict = predict, task = "class", model = "lr", ngroup = 10)
summary(cvlr)
# Performance Metrics
?mmetric
mmetric(y = bptrain$Survived, cvlr$cv.fit, metric = "ALL")
mmetric(y = bptrain$Survived, cvlr$cv.fit, metric = "CONF")
# Graphs
mgraph(y = bptrain$Survived, x = cvlr$cv.fit, graph = "ROC", TC = 0, main = "ROC Chart of Linear Regression Model", baseline = T, Grid = 10)

#################################
## Conditional Inference Trees ##
#################################

# logistic regression can handle missing data

# fit logistic regression model using fit()
?fit()
ctreem <- fit(Survived ~ ., data = bptrain, model = "ctree", task = "class")

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = ctreem, newdata = pvalid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, pvalidlabels))

# measure input importance 
ictreem <- Importance(M = ctreem, data = bptrain)
ictreem$value
# numeric vector with the computed sensitivity analysis measure for each variable
ictreem$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

# k-fold cross validation
?crossvaldata()
cvctreem <- crossvaldata(x = Survived ~. , data = bptrain, theta.fit = fit, theta.predict = predict, task = "class", model = "ctree", ngroup = 10)
summary(cvctreem)
# Performance Measures
?mmetric
mmetric(y = bptrain$Survived, cvctreem$cv.fit, metric = "ALL")
mmetric(y = bptrain$Survived, cvctreem$cv.fit, metric = "CONF")
# Graphs
mgraph(y = bptrain$Survived, x = cvctreem$cv.fit, graph = "ROC", TC = 0, main = "ROC Chart of Conditional Inference Trees Model", baseline = T, Grid = 10)


############################################
## Neural Network (Multilayer Perceptron) ##
############################################

# fit neural network model using fit()
?fit()
mlpm <- fit(Survived ~ ., data = bptrain, model = "mlp", task = "class", size = 3, decay = 0.1, maxit = 100, rang = 0.9)

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = mlpm, newdata = pvalid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, pvalidlabels))

# measure input importance 
imlpm <- Importance(M = mlpm, data = bptrain)
imlpm$value
# numeric vector with the computed sensitivity analysis measure for each variable
imlpm$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

# k-fold cross validation
?crossvaldata()
cvmlpm <- crossvaldata(x = Survived ~. , data = bptrain, theta.fit = fit, theta.predict = predict, task = "class", model = "mlp", ngroup = 10, size = 3, decay = 0.1, maxit = 100, rang=0.9)
summary(cvmlpem)
# Performance Metrics
?mmetric
mmetric(y = bptrain$Survived, cvmlpm$cv.fit, metric = "ALL")
mmetric(y = bptrain$Survived, cvmlpm$cv.fit, metric = "CONF")
# Graph
mgraph(y = bptrain$Survived, x = cvmlpm$cv.fit, graph = "ROC", TC = 0, main = "ROC Chart of Neural Networks Model", baseline = T, Grid = 10)


#####################################################
## Neural Network (Multilayer Perceptron Ensemble) ##
#####################################################

# fit neural network model using fit()
?fit()
mlpem <- fit(Survived ~ ., data = bptrain, model = "mlpe", task = "class",  size = 3, decay = 0.1, maxit = 100, rang = 0.9)

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = mlpem, newdata = pvalid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, pvalidlabels))

# measure input importance 
imlpem <- Importance(M = mlpem, data = bptrain)
imlpem$value
# numeric vector with the computed sensitivity analysis measure for each variable
imlpem$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

# k-fold cross validation
?crossvaldata()
cvmlpem <- crossvaldata(x = Survived ~. , data = bptrain, theta.fit = fit, theta.predict = predict, task = "class", model = "mlpe", ngroup = 10,  size = 3, decay = 0.1, maxit = 100, rang = 0.9)
summary(cvmlpem)
?mmetric
mmetric(y = bptrain$Survived, cvmlpem$cv.fit, metric = "ALL")
mmetric(y = bptrain$Survived, cvmlpem$cv.fit, metric = "CONF")

###############################
## eXtreme Gradient Boosting ##
###############################

# fit neural network model using fit()
?fit()
xgbm <- fit(Survived ~ ., data = bptrain, model = "xgboost", task = "class")

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = xgbm, newdata = pvalid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, pvalidlabels))

# measure input importance 
ixgbm <- Importance(M = xgbm, data = bptrain)
ixgbm$value
# numeric vector with the computed sensitivity analysis measure for each variable
ixgbm$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

# k-fold cross validation
?crossvaldata()
cvxgbm <- crossvaldata(x = Survived ~. , data = bptrain, theta.fit = fit, theta.predict = predict, task = "class", model = "xgboost", ngroup = 10)
summary(cvxgbm)
?mmetric
mmetric(y = bptrain$Survived, cvxgbm$cv.fit, metric = "ALL")
mmetric(y = bptrain$Survived, cvxgbm$cv.fit, metric = "CONF")

#############
## Bagging ##
#############

# fit neural network model using fit()
?fit()
bggm <- fit(Survived ~ ., data = bptrain, model = "bagging", task = "class")

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = bggm, newdata = pvalid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, pvalidlabels))

# measure input importance 
ibggm <- Importance(M = bggm, data = bptrain)
ibggm$value
# numeric vector with the computed sensitivity analysis measure for each variable
ibggm$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

# k-fold cross validation
?crossvaldata()
cvbggm <- crossvaldata(x = Survived ~. , data = bptrain, theta.fit = fit, theta.predict = predict, task = "class", model = "bagging", ngroup = 10)
summary(cvbggm)
?mmetric
mmetric(y = bptrain$Survived, cvbggm$cv.fit, metric = "ALL")
mmetric(y = bptrain$Survived, cvbggm$cv.fit, metric = "CONF")

##############
## Boosting ##
##############

# fit neural network model using fit()
?fit()
bogm <- fit(Survived ~ ., data = bptrain, model = "bagging", task = "class")

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = bogm, newdata = pvalid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, pvalidlabels))

# measure input importance 
ibogm <- Importance(M = bogm, data = bptrain)
ibogm$value
# numeric vector with the computed sensitivity analysis measure for each variable
ibogm$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

# k-fold cross validation
?crossvaldata()
cvbgom <- crossvaldata(x = Survived ~. , data = bptrain, theta.fit = fit, theta.predict = predict, task = "class", model = "boosting", 10)
summary(cvbogm)
?mmetric()
mmetric(y = bptrain$Survived, cvbogm$cv.fit, metric = "ALL")
mmetric(y = bptrain$Survived, cvbogm$cv.fit, metric = "CONF")

##################################
## Linear Discriminant Analysis ##
##################################

# fit neural network model using fit()
?fit()
ldam <- fit(Survived ~ ., data = bptrain, model = "lda", task = "class")

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = ldam, newdata = pvalid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, pvalidlabels))

# measure input importance 
ildam <- Importance(M = ldam, data = bptrain)
ildam$value
# numeric vector with the computed sensitivity analysis measure for each variable
ildam$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

# k-fold cross validation
?crossvaldata()
cvldam <- crossvaldata(x = Survived ~. , data = bptrain, theta.fit = fit, theta.predict = predict, task = "class", model = "lda", ngroup = 10)
summary(cvldam)
?mmetric
mmetric(y = bptrain$Survived, cvldam$cv.fit, metric = "ALL")
mmetric(y = bptrain$Survived, cvldam$cv.fit, metric = "CONF")

############################
## Naivebayes Classifiers ##
############################

# fit neural network model using fit()
?fit()
nbmm <- fit(Survived ~ ., data = bptrain, model = "naivebayes", task = "class", laplace = 0)

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = nbmm, newdata = pvalid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, pvalidlabels))

# measure input importance 
inbmm <- Importance(M = nbmm, data = bptrain)
inbmm$value
# numeric vector with the computed sensitivity analysis measure for each variable
inbmm$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

# k-fold cross validation
?crossvaldata()
cvnbmm <- crossvaldata(x = Survived ~. , data = bptrain, theta.fit = fit, theta.predict = predict, task = "class", model = "naivebayes", ngroup = 10, laplace = 0)
summary(cvnbmm)
# Performance Metrics
?mmetric
mmetric(y = bptrain$Survived, cvnbmm$cv.fit, metric = "ALL")
mmetric(y = bptrain$Survived, cvnbmm$cv.fit, metric = "CONF")
# Graphs
mgraph(y = bptrain$Survived, x = cvnbmm$cv.fit, graph = "ROC", TC = 0, main = "ROC Chart of Naive Bayes Classifiers Model", baseline = T, Grid = 10)


#####################################
## Quadratic Discriminant Analysis ##
#####################################

# fit neural network model using fit()
?fit()
qdam <- fit(Survived ~ ., data = bptrain, model = "qda", task = "class")

# generate predictions for model using validation data 
?predict.fit
pred <- predict(object = qdam, newdata = pvalid, type = "class")

# construct confusion matrix 
?confusionMatrix()
confusionMatrix(table(pred, pvalidlabels))

# measure input importance 
iqdam <- Importance(M = qdam, data = bptrain)
iqdam$value
# numeric vector with the computed sensitivity analysis measure for each variable
iqdam$imp
# numeric vector with the relative importance for each variable
# emarked is the most important variable relative to the others @ 47% importance

# k-fold cross validation
?crossvaldata()
cqdam <- crossvaldata(x = Survived ~. , data = bptrain, theta.fit = fit, theta.predict = predict, task = "class", model = "qda", ngroup = 10)
summary(cvqdam)
?mmetric
mmetric(y = bptrain$Survived, cvqdam$cv.fit, metric = "ALL")
mmetric(y = bptrain$Survived, cvqdam$cv.fit, metric = "CONF")


##############################################################################################################################
## Evaluation Results ########################################################################################################
##############################################################################################################################

# K-Cross Fold Evaluation results
modelnames <- c("Decision Tree", "Random Forests", "Bayesian Network", "General Linear Model", "K-Nearest Neighbours", "Support Vector Machines", "Logistic Regression", "Conditional Inference Trees", "NN Multi-layer Perceptron", "NN Multi-layer Perceptron Ensemble", "Extreme Gradient Boosting", "Naive Bayes")
evalresults <- rbind(mmetric(y = bptrain$Survived, cvdt$cv.fit, metric = "ALL"),
                     mmetric(y = bptrain$Survived, cvrf$cv.fit, metric = "ALL"),
                     mmetric(y = bptrain$Survived, cvnb$cv.fit, metric = "ALL"),
                     mmetric(y = bptrain$Survived, cvcvglm$cv.fit, metric = "ALL"),
                     mmetric(y = bptrain$Survived, cvknn$cv.fit, metric = "ALL"),
                     mmetric(y = bptrain$Survived, cvsvm$cv.fit, metric = "ALL"),
                     mmetric(y = bptrain$Survived, cvlr$cv.fit, metric = "ALL"),
                     mmetric(y = bptrain$Survived, cvctreem$cv.fit, metric = "ALL"),
                     mmetric(y = bptrain$Survived, cvmlpm$cv.fit, metric = "ALL"),
                     mmetric(y = bptrain$Survived, cvmlpem$cv.fit, metric = "ALL"),
                     mmetric(y = bptrain$Survived, cvxgbm$cv.fit, metric = "ALL"),
                     mmetric(y = bptrain$Survived, cvnbmm$cv.fit, metric = "ALL")
)
evalresults
# save evalresults to a csv text file
write.table(evalresults, file = "evalresults2.csv", sep = ",", quote = F, row.names = modelnames)

##############################################################################################################################
## Predicting the Test Set ###################################################################################################
##############################################################################################################################

# Option 1: predict with one model
# Option 2: predict with an ensemble model

View(ptest)

dtreepred <- predict(object = dtree, newdata = ptest, type = "class")
rforestpred <- predict(object = rforest, newdata = ptest, type = "class")
nbayespred <- predict(object = nbayes, newdata = ptest, type = "class")
cvglmpred <- predict(object = cvglm, newdata = ptest, type = "class")
knnmpred <- predict(object = knnm, newdata = ptest, type = "class")
svmmpred <- predict(object = svmm, newdata = ptest, type = "class")
lrpred <- predict(object = lr, newdata = ptest, ptype = "class")
ctreempred <- predict(object = ctreem, newdata = ptest, type = "class")
mlpmpred <- predict(object = mlpm, newdata = ptest, type = "class")
mlpempred <- predict(object = mlpem, newdata = ptest, type = "class")
xgbmpred <- predict(object = xgbm, newdata = ptest, type = "class")
naivebayespred <- predict(object = nbmm, newdata = ptest, type = "class")

###########################################################################################################################
## Esemble Models ####################################################################################################
###########################################################################################################################

# method 1: intersection of models
# step 1 use dtree
testpred <- dtreepred
# step 2 fill in common predictions from dtree, svm and knn (multiple models)
length(knnmpred)
length(x = dtreepred)
length(x = svmmpred)
i1 <- which(dtreepred == rforestpred)
i2 <- which(svmmpred == lrpred)
i <- intersect(i1, i2)
testpred[i] <- svmmpred[i]

# method 2: data frame of predictions and choose most common
ensemblepredictions <- cbind(dtreepred, rforestpred, knnmpred, svmmpred, lrpred, ctreempred, mlpmpred, mlpempred, naivebayespred)
ensemblepredictions <- ifelse(test = ensemblepredictions == 1, yes = 0, no = 1)
PassengerId <- c(892:1309)
ensemblepredictions <- cbind(PassengerId, ensemblepredictions)
# save ensemble predictions to a csv text file
write.table(ensemblepredictions, file = "ensemblepredictions.csv", sep = ",", quote = F)
# find must common values in the table
?tabulate()
m <- tabulate(ensemblepredictions[1, ], nbins = 2)
for(i in 2:nrow(ensemblepredictions)) {
  m <- rbind(m, tabulate(ensemblepredictions[i, ], nbins = 2))
}
m <- as.data.frame(m)
colnames(m) <- c("0","1")
f <- 1
for(i in 2:nrow(m)){
  f <- rbind(f, as.integer(which.max(x = m[i,])))
}
f
testpred <- ifelse(test = f == 1, yes = 0, no = 1)
Survived <- testpred

# single models
testpred <- dtreepred
testpred <- rforestpred
testpred <- svmmpred
testpred <- knnmpred
###########################################################################################################################
## Writing to CSV files ###################################################################################################
###########################################################################################################################

head(testpred)
length(testpred)
testpred <- as.numeric(x = testpred)
Survived <- ifelse(test = testpred %in% 1, yes = 0, no = 1)
PassengerId <- c(892:1309)
length(PassengerId)
testpreddf <- cbind(PassengerId, Survived)
head(testpreddf)
?write.table()
write.table(testpreddf, file = "testpred20.csv", sep = ",", quote = F, row.names = F, col.names = T)
