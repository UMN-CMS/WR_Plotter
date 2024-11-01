import os,ROOT
from Plotter import SampleGroup

## DY
SampleGroup_DY_2018 = SampleGroup(
  Name='DY',
  Type='Bkgd',
  Samples=['DYJets'],
  Year = 2018,
  Color='#FFFF00',
  Style=1,
  TLatexAlias='Z+jets',
  LatexAlias='ZJets'
)

## ttbar
SampleGroup_TT_TW_2018 = SampleGroup(
  Name='TT_TW',
  Type='Bkgd',
  Samples=['tt+tW'],
  Year = 2018,
  Color='#FF0000',
  Style=1,
  TLatexAlias=r'$t\bar{t}+tW$',
  LatexAlias='TT\_TW'
)

## NonPrompt
SampleGroup_NonPrompt_2018 = SampleGroup(
  Name='NonPrompt',
  Type='Bkgd',
  Samples=['Nonprompt'],
  Year = 2018,
  Color='#32CD32',
  Style=1,
  TLatexAlias='Nonprompt',
  LatexAlias='NonPrompt'
)

## others
SampleGroup_Others_2018 = SampleGroup(
  Name='Other',
  Type='Bkgd',
  Samples=['Other'],
  Year = 2018,
  Color='#00BFFF',
  Style=1,
  TLatexAlias='Other backgrounds',
  LatexAlias='Others'
)
