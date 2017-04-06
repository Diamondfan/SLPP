#!/bin/bash
#Author:Richardfan
#2017.3.21

cd ../
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh

n=8

#corpus and trans directory
corpus="/home/Audio_Data/863_corpus/"

mono_test_ali=prepare_for_LPP/mono_test_ali
if [ ! -d $mono_test_ali ]; then
	mkdir -p $mono_test_ali
fi

#steps/align_si.sh --boost-silence 1.25 --nj $n --cmd "$train_cmd" \
#		data/test data/lang exp/mono $mono_test_ali || exit 1;

for x in train test; do
	#make  mfcc_13
   	feat_scp=data/$x/feats.scp
	data=data/$x
    #copy-feats scp:$feat_scp ark,t:prepare_for_LPP/${x}_mfcc_13.txt
   
	#make mfcc_39
	#apply-cmvn --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp \
	#	scp:$data/feats.scp ark:- | add-deltas ark:- \
	#   	ark,t:prepare_for_LPP/${x}_mfcc_39.txt

	#make mfcc_13 for concatenating n frames
	apply-cmvn --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp \
		scp:$feat_scp ark:- | splice-feats --left-context=4 \
		--right-context=4 ark:- ark,t:prepare_for_LPP/${x}_mfcc_frame9.txt
	sed -i 's:\[::g' prepare_for_LPP/${x}_mfcc_frame9.txt
	sed -i 's: \]::g' prepare_for_LPP/${x}_mfcc_frame9.txt
	sed -i '/^[FM].*/d' prepare_for_LPP/${x}_mfcc_frame9.txt
done
<<!
for i in 1 2 3 4 5 6 7 8; do
	gunzip -c exp/mono_ali/ali.$i.gz | ../../../src/bin/ali-to-pdf exp/mono_ali/final.mdl ark:- ark,t:tmp.txt
	cat tmp.txt >> prepare_for_LPP/train_pdf.txt
	gunzip -c $mono_test_ali/ali.$i.gz | ../../../src/bin/ali-to-pdf $mono_test_ali/final.mdl ark:- ark,t:tmp.txt
	cat tmp.txt >> prepare_for_LPP/test_pdf.txt
done
rm -f tmp.txt
!
