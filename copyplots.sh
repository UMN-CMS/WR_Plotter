MASS_OPTIONS=(
  WR1200_N200 WR1200_N400 WR1200_N600 WR1200_N800 WR1200_N1100
  WR1600_N400 WR1600_N600 WR1600_N800 WR1600_N1200 WR1600_N1500
  WR2000_N400 WR2000_N800 WR2000_N1000 WR2000_N1400 WR2000_N1900
  WR2400_N600 WR2400_N800 WR2400_N1200 WR2400_N1800 WR2400_N2300
  WR2800_N600 WR2800_N1000 WR2800_N1400 WR2800_N2000 WR2800_N2700
  WR3200_N800 WR3200_N1200 WR3200_N1600 WR3200_N2400 WR3200_N3000
)
mkdir relevant
cd plots
for mass in "${MASS_OPTIONS[@]}"; do
  mkdir ../relevant/"${mass}"
  mkdir ../relevant/"${mass}"/ee
  mkdir ../relevant/"${mass}"/mumu
  cd "${mass}"/WR_EE_Resolved_SR
  cp ComparePTnorm_WR_EE_Resolved_SR.pdf ../../../relevant/"${mass}"/ee/ptnorm.pdf
  cp ComparePT_WR_EE_Resolved_SR.pdf ../../../relevant/"${mass}"/ee/ptrel.pdf
  cp CompareSine_WR_EE_Resolved_SR.pdf ../../../relevant/"${mass}"/ee/sine.pdf
  cd ..
  cd WR_MuMu_Resolved_SR
  cp ComparePTnorm_WR_MuMu_Resolved_SR.pdf ../../../relevant/"${mass}"/mumu/ptnorm.pdf
  cp ComparePT_WR_MuMu_Resolved_SR.pdf ../../../relevant/"${mass}"/mumu/ptrel.pdf
  cp CompareSine_WR_MuMu_Resolved_SR.pdf ../../../relevant/"${mass}"/mumu/sine.pdf
  cd ..
  cd ..
done
cd ..
