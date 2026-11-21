#!/usr/bin/env perl
#
# Split a list of wave files into a training and a test list.
#
# Usage: create_yesno_waves_test_train.pl <full-list> <test-list> <train-list> [<train-count>]
#
# The first <train-count> entries go to training and the rest to test. The
# count defaults to half the list, rounded down, which is what this script
# always did; the caller states it explicitly where a particular split has to
# be preserved. See the comment in ../prepare_data.sh: yesno's published
# results were measured on a 31/29 split that arose from two README files
# being counted as waves, and that split is worth keeping after they stopped
# being counted.

$full_list = $ARGV[0];
$test_list = $ARGV[1];
$train_list = $ARGV[2];
$train_count = $ARGV[3];

open FL, $full_list;
$nol = 0;
while ($l = <FL>)
{
	$nol++;
}
close FL;

if (!defined $train_count || $train_count eq "")
{
	$train_count = int($nol / 2);
}
if ($train_count > $nol)
{
	die "$0: asked for $train_count training entries but the list has only $nol\n";
}

$i = 0;
open FL, $full_list;
open TESTLIST, ">$test_list";
open TRAINLIST, ">$train_list";
while ($l = <FL>)
{
	chomp($l);
	$i++;
	if ($i <= $train_count )
	{
		print TRAINLIST "$l\n";
	}
	else
	{
		print TESTLIST "$l\n";
	}
}
