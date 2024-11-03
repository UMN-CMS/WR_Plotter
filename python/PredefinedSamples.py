import os,ROOT
from Plotter import SampleGroup

## DY
SampleGroup_DY_2018 = SampleGroup(
  name='DY',
  Type='Bkgd',
  samples=['DYJets'],
  year = 2018,
  color='#5790fc',
  style=1,
  tlatex_alias='Z+jets',
  latex_alias='ZJets'
)

## ttbar
SampleGroup_TT_TW_2018 = SampleGroup(
  name='TT_TW',
  Type='Bkgd',
  samples=['tt+tW'],
  year = 2018,
  color='#f89c20',
  style=1,
  tlatex_alias=r'$t\bar{t}+tW$',
  latex_alias='TT\_TW'
)

## NonPrompt
SampleGroup_NonPrompt_2018 = SampleGroup(
  name='NonPrompt',
  Type='Bkgd',
  samples=['Nonprompt'],
  year = 2018,
  color='#e42536',
  style=1,
  tlatex_alias='Nonprompt',
  latex_alias='NonPrompt'
)

## others
SampleGroup_Others_2018 = SampleGroup(
  name='Other',
  Type='Bkgd',
  samples=['Other'],
  year = 2018,
  color='#964a8b',
  style=1,
  tlatex_alias='Other backgrounds',
  latex_alias='Others'
)
