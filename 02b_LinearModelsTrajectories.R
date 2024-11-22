#=========================================================
# JOINT DRAWING EXPERIMENT
#
# The script performs the main linear modeling on the
# visuomotor interference measure (boxcox corrected area).
# Additionally, it performs the two control analyses
# (movement duration and task order)
#=========================================================


library(sjPlot) #for plotting lmer and glmer mods
library(fitdistrplus)
library(tidyverse) #for all data wrangling
library(cowplot) #for manuscript ready figures
library(lme4) #for lmer & glmer models
library(sjmisc) 
library(effects)
library(sjstats) #use for r2 functions
library(car)
library(DHARMa)
library(ggpubr)
library(lmerTest)
library(report)

# Loading data
data <- read.csv("Distances_clean.csv", stringsAsFactors=T)

#Setting sum contrast
options(contrasts = c("contr.sum", "contr.sum"))
contrasts(data$task)
contrasts(data$cong)

ggqqplot(data$boxcox)
hist(data$boxcox)
descdist(data$boxcox)

#perform shapiro-wilk test
shapiro.test(sample(data$boxcox, 1000))

#=========================================================
# Model selection
#=========================================================

mod0 = lmer(boxcox ~ task*cong + (1|Subject), data = data)
#mod1 = lmer(boxcox ~ task*cong + (1+task|Subject), data = data)
#mod2 = lmer(boxcox ~ task*cong + (1+cong|Subject), data = data)
#mod3 = lmer(boxcox ~ task*cong + (1+task+cong|Subject), data = data)
#mod4 = lmer(boxcox ~ task*cong + (1+task*cong|Subject), data = data)
#mod5 = lmer(boxcox ~ task*cong + (1+task:cong|Subject), data = data)

summary(mod0)
anova(mod0)
coef(mod0)

plot(effect('task:cong', mod0), xlab = "task", ylab = "area", main = "interaction")
plot_model(mod0, transform = NULL,type = "re")
plot_model(mod0, transform = NULL,type = "diag", terms = c('task', 'cong'))
qqnorm(resid(mod0))
qqline(resid(mod0))
simulationOutput <- simulateResiduals(fittedModel = mod0, plot = T)
plot(density(resid(mod0)))


ranova(mod0)
report(mod0)




#=========================================================
# Control analysis 1
# Duration of the participant's movement
#=========================================================

mod_mov_dur <- lmer(boxcox ~ task*cong*scale(pp_mov_dur) + (1|Subject), data = data)
summary(mod_mov_dur)
plot(effect('cong:pp_mov_dur', mod_mov_dur),  ylab = "area", main = "pp_mov_dur")
plot(effect('task:pp_mov_dur', mod_mov_dur),  ylab = "area", main = "pp_mov_dur")

report(mod_mov_dur)
# The results show an interaction of congruency and pp_mov_dur
# but in the opposite direction
# for incongruent trials, slower trials are also those with a larger interference
# so it cannot be an effect of speed-accuracy trade-off


#=========================================================
# Control analysis 2
# Task Order
#=========================================================

# Loading data
data_to <- read.csv("test_taskorder.csv", stringsAsFactors=T)

#Setting sum contrast
options(contrasts = c("contr.sum", "contr.sum"))
contrasts(data_to$task)
contrasts(data_to$cong)
contrasts(data_to$TaskOrder)

mod_TO <- lmer(boxcox ~ task*cong*TaskOrder + (1|Subject), data = data_to)
summary(mod_TO)
plot(effect('task:cong:TaskOrder', mod_TO),  ylab = "area")

## Task ORDER does not interact with anything
