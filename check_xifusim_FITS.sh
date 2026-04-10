#sim=3
sim=$1
fluxes=("flux0.32mcrab" "flux0.50mcrab" "flux1.00mcrab" "flux3.20mcrab" "flux10.00mcrab" "flux31.60mcrab" "flux100.00mcrab" "flux316.20mcrab" "flux1000.00mcrab")
date1=`date +%Y%m%d`
date2=`date +%H%M%S`
report_file="checkFITS_sim${sim}.txt"
echo "Checking FITS files of data in ${fluxdir} and sim ${sim} on DAY=$date1 TIME=$date2:" >${report_file}

for flux in ${fluxes[@]}; do
    fluxdir="v5_20250621/${flux}"
    files=`ls ${fluxdir}/sim_${sim}/*_xifusim.fits`

    for file in $files
    do
	#echo "Checking $file"
	ftverify $file prstat=NO
	numerrs=`pget ftverify numerrs`
	if [[ $numerrs -gt 0 ]]; then
	    echo "      $file is corrupted" >> ${report_file}
	fi
    done
done
