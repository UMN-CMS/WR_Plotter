import os,ROOT
from Plotter import SampleGroup

## DY
SampleGroup_DY_2018 = SampleGroup(
  name='DY',
  mc_campaign='Run2Summer20UL18',
  year = 2018,
  color='#5790fc',
  samples=['DYJets'],
  tlatex_alias='Z+jets',
)

## ttbar
#SampleGroup_TT_TW_2018 = SampleGroup(
#  name='TT_TW',
#  mc_campaign='Run2Summer20UL18',
#  year = 2018,
#  color='#f89c20',
#  samples=['tt+tW'],
#  tlatex_alias=r'$t\bar{t}+tW$',
#)

## ttbar
SampleGroup_TT_2018 = SampleGroup(
  name='TT',
  mc_campaign='Run2Summer20UL18',
  year = 2018,
  color='#f89c20',
  samples=['tt'],
  tlatex_alias=r'$t\bar{t}$',
)

## NonPrompt
SampleGroup_NonPrompt_2018 = SampleGroup(
  name='NonPrompt',
  mc_campaign='Run2Summer20UL18',
  year = 2018,
  color='#e42536',
  samples=['Nonprompt'],
  tlatex_alias='Nonprompt',
)

## others
SampleGroup_Others_2018 = SampleGroup(
  name='Other',
  mc_campaign='Run2Summer20UL18',
  year = 2018,
  color='#964a8b',
  samples=['Other'],
  tlatex_alias='Other backgrounds',
)
