from .plotter import SampleGroup

SampleGroup_RunIISummer20UL18_DY = SampleGroup(
  name='DY',
  run='RunII',
  year = 2018,
  mc_campaign='RunIISummer20UL18',
  color='#5790fc',
  samples=['DYJets'],
  tlatex_alias='Z+jets',
)

SampleGroup_RunIISummer20UL18_TTbar = SampleGroup(
  name='tt',
  run='RunII',
  year = 2018,
  mc_campaign='RunIISummer20UL18',
  color='#f89c20',
  samples=['TTbar', 'tW'],
  tlatex_alias=r'$t\bar{t}+tW$',
)

SampleGroup_RunIISummer20UL18_Nonprompt = SampleGroup(
  name='NonPrompt',
  run='RunII',
  year = 2018,
  mc_campaign='RunIISummer20UL18',
  color='#e42536',
  samples=['WJets', 'SingleTop', 'TTbarSemileptonic'],
  tlatex_alias='Nonprompt',
)

SampleGroup_RunIISummer20UL18_Other = SampleGroup(
  name='Other',
  run='RunII',
  year = 2018,
  mc_campaign='RunIISummer20UL18',
  color='#964a8b',
  samples=['TTX', 'Diboson', 'Triboson'],
  tlatex_alias='Other backgrounds',
)

SampleGroup_RunIISummer20UL18_EGamma = SampleGroup(
  name='EGamma',
  run='RunII',
  year = 2018,
  mc_campaign='RunIISummer20UL18',
  color='#964a8b',
  samples=['EGamma'],
  tlatex_alias='Data',
)

SampleGroup_RunIISummer20UL18_Muon = SampleGroup(
  name='Muon',
  run='RunII',
  year = 2018,
  mc_campaign='RunIISummer20UL18',
  color='#964a8b',
  samples=['Muon'],
  tlatex_alias='Data',
)

SampleGroup_Run3Summer22_DY = SampleGroup(
  name='DY',
  run='Run3',
  year = 2022,
  mc_campaign='Run3Summer22',
  color='#5790fc',
  samples=['DYJets'],
  tlatex_alias='Z+jets',
)

SampleGroup_Run3Summer22_TTbar = SampleGroup(
  name='tt',
  run='Run3',
  year = 2022,
  mc_campaign='Run3Summer22',
  color='#f89c20',
  samples=['TTbar', 'tW'],
  tlatex_alias=r'$t\bar{t}+tW$',
)

SampleGroup_Run3Summer22_Nonprompt = SampleGroup(
  name='NonPrompt',
  run='Run3',
  year = 2022,
  mc_campaign='Run3Summer22',
  color='#e42536',
  samples=['WJets', 'SingleTop', 'TTbarSemileptonic'],
  tlatex_alias='Nonprompt',
)

SampleGroup_Run3Summer22_Other = SampleGroup(
  name='Other',
  run='Run3',
  year = 2022,
  mc_campaign='Run3Summer22',
  color='#964a8b',
  samples=['TTX', 'Diboson', 'Triboson'],
  tlatex_alias='Other backgrounds',
)

SampleGroup_Run3Summer22_EGamma = SampleGroup(
  name='EGamma',
  run='Run3',
  year = 2022,
  mc_campaign='Run3Summer22',
  color='#964a8b',
  samples=['EGamma'],
  tlatex_alias='Data',
)

SampleGroup_Run3Summer22_Muon = SampleGroup(
  name='Muon',
  run='Run3',
  year = 2022,
  mc_campaign='Run3Summer22',
  color='#964a8b',
  samples=['Muon'],
  tlatex_alias='Data',
)

SampleGroup_Run3Summer22EE_DY = SampleGroup(
  name='DY',
  run='Run3',
  year = 2022,
  mc_campaign='Run3Summer22EE',
  color='#5790fc',
  samples=['DYJets'],
  tlatex_alias='Z+jets',
)

SampleGroup_Run3Summer22EE_TTbar = SampleGroup(
  name='tt',
  run='Run3',
  year = 2022,
  mc_campaign='Run3Summer22EE',
  color='#f89c20',
  samples=['TTbar', 'tW'],
  tlatex_alias=r'$t\bar{t}+tW$',
)

SampleGroup_Run3Summer22EE_Nonprompt = SampleGroup(
  name='NonPrompt',
  run='Run3',
  year = 2022,
  mc_campaign='Run3Summer22EE',
  color='#e42536',
  samples=['WJets', 'SingleTop', 'TTbarSemileptonic'],
  tlatex_alias='Nonprompt',
)

SampleGroup_Run3Summer22EE_Other = SampleGroup(
  name='Other',
  run='Run3',
  year = 2022,
  mc_campaign='Run3Summer22EE',
  color='#964a8b',
  samples=['TTX', 'Diboson', 'Triboson'],
  tlatex_alias='Other backgrounds',
)

SampleGroup_Run3Summer22EE_EGamma = SampleGroup(
  name='EGamma',
  run='Run3',
  year = 2022,
  mc_campaign='Run3Summer22EE',
  color='#964a8b',
  samples=['Run2018E', 'Run2018F', 'Run2018G'],
  tlatex_alias='Data',
)

SampleGroup_Run3Summer22EE_SingleMuon = SampleGroup(
  name='SingleMuon',
  run='Run3',
  year = 2022,
  mc_campaign='Run3Summer22EE',
  color='#964a8b',
  samples=['Run2018E', 'Run2018F', 'Run2018G'],
  tlatex_alias='Data',
)
