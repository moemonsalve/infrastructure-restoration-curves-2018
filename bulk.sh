rm out1.pdf
rm out2.pdf
rm res1.pdf
rm res2.pdf

rm -rf _bulk_$1
mkdir _bulk_$1
cd $1

for x in ok-*.csv; do

  echo $x

  python ../plot-it.py $x
  mv out.pdf ../_bulk_$1/pl-$x.pdf

  #python ../inter-est.py --meanfield $x
  #mv out.pdf ../_bulk_$1/mf-$x.pdf

  #python ../inter-est.py --stochastic $x
  #mv out.pdf ../_bulk_$1/st-$x.pdf

  python ../inter-est3.py $x > ../_bulk_$1/txt-$x.txt
  pdfcrop -margins '0 0 0 3' out1.pdf
  pdfcrop -margins '0 0 0 3' out2.pdf
  pdfcrop -margins '0 0 0 3' res1.pdf
  pdfcrop -margins '0 0 0 3' res2.pdf
  mv out1-crop.pdf ../_bulk_$1/mf-$x.pdf
  mv out2-crop.pdf ../_bulk_$1/st-$x.pdf
  mv res1-crop.pdf ../_bulk_$1/r1-$x.pdf
  mv res2-crop.pdf ../_bulk_$1/r2-$x.pdf
  rm out1.pdf out2.pdf res1.pdf res2.pdf
  
  python ../vs-iio.py $x > ../_bulk_$1/vio-$x.txt
  pdfcrop -margins '0 0 0 3' iio.pdf
  pdfcrop -margins '0 0 0 3' r3.pdf
  pdfcrop -margins '0 0 0 3' r4.pdf
  pdfcrop -margins '0 0 0 3' sss.pdf
  mv iio-crop.pdf  ../_bulk_$1/io-$x.pdf
  mv sss-crop.pdf  ../_bulk_$1/ss-$x.pdf
  mv r3-crop.pdf  ../_bulk_$1/r3-$x.pdf
  mv r4-crop.pdf  ../_bulk_$1/r4-$x.pdf
  rm iio.pdf sss.pdf r3.pdf r4.pdf

  python ../inter-est.py --latex $x > ../_bulk_$1/tex-$x.txt

done

cd ..

python _simplify_names.py _bulk_$1
cat _bulk_$1/vio-*.txt > _bulk_$1/vio-ALL.txt
