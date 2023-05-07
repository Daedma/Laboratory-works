	.text
	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.file	"main.cc"
	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@3ff0000000000000         # -- Begin function main
	.section	.rdata,"dr",discard,__real@3ff0000000000000
	.p2align	3
__real@3ff0000000000000:
	.quad	0x3ff0000000000000              # double 1
	.globl	__real@403c000000000000
	.section	.rdata,"dr",discard,__real@403c000000000000
	.p2align	3
__real@403c000000000000:
	.quad	0x403c000000000000              # double 28
	.globl	__real@4010000000000000
	.section	.rdata,"dr",discard,__real@4010000000000000
	.p2align	3
__real@4010000000000000:
	.quad	0x4010000000000000              # double 4
	.text
	.globl	main
	.p2align	4, 0x90
main:                                   # @main
# %bb.0:
	movsd	"?c@@3NA"(%rip), %xmm1          # xmm1 = mem[0],zero
	movsd	__real@4010000000000000(%rip), %xmm0 # xmm0 = mem[0],zero
	divsd	%xmm0, %xmm1
	movsd	"?d@@3NA"(%rip), %xmm0          # xmm0 = mem[0],zero
	movsd	__real@403c000000000000(%rip), %xmm2 # xmm2 = mem[0],zero
	mulsd	%xmm2, %xmm0
	addsd	%xmm1, %xmm0
	movsd	"?a@@3NA"(%rip), %xmm1          # xmm1 = mem[0],zero
	divsd	"?d@@3NA"(%rip), %xmm1
	subsd	"?c@@3NA"(%rip), %xmm1
	movsd	__real@3ff0000000000000(%rip), %xmm2 # xmm2 = mem[0],zero
	subsd	%xmm2, %xmm1
	divsd	%xmm1, %xmm0
	movsd	%xmm0, "?result@@3NA"(%rip)
	xorl	%eax, %eax
	retq
                                        # -- End function
	.bss
	.globl	"?a@@3NA"                       # @"?a@@3NA"
	.p2align	3
"?a@@3NA":
	.quad	0x0000000000000000              # double 0

	.globl	"?d@@3NA"                       # @"?d@@3NA"
	.p2align	3
"?d@@3NA":
	.quad	0x0000000000000000              # double 0

	.globl	"?c@@3NA"                       # @"?c@@3NA"
	.p2align	3
"?c@@3NA":
	.quad	0x0000000000000000              # double 0

	.globl	"?result@@3NA"                  # @"?result@@3NA"
	.p2align	3
"?result@@3NA":
	.quad	0x0000000000000000              # double 0

	.addrsig
	.addrsig_sym "?a@@3NA"
	.addrsig_sym "?d@@3NA"
	.addrsig_sym "?c@@3NA"
	.addrsig_sym "?result@@3NA"
	.globl	_fltused
