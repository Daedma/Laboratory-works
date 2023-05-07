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
	.globl	__real@3fd0000000000000         # -- Begin function main
	.section	.rdata,"dr",discard,__real@3fd0000000000000
	.p2align	3
__real@3fd0000000000000:
	.quad	0x3fd0000000000000              # double 0.25
	.globl	__real@403c000000000000
	.section	.rdata,"dr",discard,__real@403c000000000000
	.p2align	3
__real@403c000000000000:
	.quad	0x403c000000000000              # double 28
	.globl	__real@bff0000000000000
	.section	.rdata,"dr",discard,__real@bff0000000000000
	.p2align	3
__real@bff0000000000000:
	.quad	0xbff0000000000000              # double -1
	.text
	.globl	main
	.p2align	4, 0x90
main:                                   # @main
# %bb.0:
	movsd	"?c@@3NA"(%rip), %xmm0          # xmm0 = mem[0],zero
	movsd	"?d@@3NA"(%rip), %xmm1          # xmm1 = mem[0],zero
	movsd	__real@403c000000000000(%rip), %xmm2 # xmm2 = mem[0],zero
	mulsd	%xmm1, %xmm2
	movsd	"?a@@3NA"(%rip), %xmm3          # xmm3 = mem[0],zero
	divsd	%xmm1, %xmm3
	movsd	__real@3fd0000000000000(%rip), %xmm1 # xmm1 = mem[0],zero
	mulsd	%xmm0, %xmm1
	subsd	%xmm0, %xmm3
	addsd	__real@bff0000000000000(%rip), %xmm3
	addsd	%xmm1, %xmm2
	divsd	%xmm3, %xmm2
	movsd	%xmm2, "?result@@3NA"(%rip)
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
	.globl	_fltused
