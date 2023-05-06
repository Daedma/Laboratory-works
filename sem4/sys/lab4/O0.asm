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
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
main:                                   # @main
# %bb.0:
	movl	"?size@@3HA"(%rip), %r9d
	testq	%r9, %r9
	je	.LBB0_6
# %bb.1:
	movq	"?a@@3PEAHEA"(%rip), %r10
	movl	"?d@@3HA"(%rip), %r8d
	xorl	%edx, %edx
	jmp	.LBB0_2
	.p2align	4, 0x90
.LBB0_5:                                #   in Loop: Header=BB0_2 Depth=1
	incq	%rdx
	cmpq	%rdx, %r9
	je	.LBB0_6
.LBB0_2:                                # =>This Inner Loop Header: Depth=1
	movl	(%r10,%rdx,4), %ecx
	testl	%ecx, %ecx
	jns	.LBB0_5
# %bb.3:                                #   in Loop: Header=BB0_2 Depth=1
	cmpl	%r8d, %ecx
	jge	.LBB0_5
# %bb.4:                                #   in Loop: Header=BB0_2 Depth=1
	movl	%ecx, %eax
	imull	%ecx, %eax
	imull	%ecx, %eax
	addl	%eax, "?res@@3HA"(%rip)
	jmp	.LBB0_5
.LBB0_6:
	xorl	%eax, %eax
	retq
                                        # -- End function
	.bss
	.globl	"?res@@3HA"                     # @"?res@@3HA"
	.p2align	2
"?res@@3HA":
	.long	0                               # 0x0

	.globl	"?d@@3HA"                       # @"?d@@3HA"
	.p2align	2
"?d@@3HA":
	.long	0                               # 0x0

	.globl	"?size@@3HA"                    # @"?size@@3HA"
	.p2align	2
"?size@@3HA":
	.long	0                               # 0x0

	.globl	"?a@@3PEAHEA"                   # @"?a@@3PEAHEA"
	.p2align	3
"?a@@3PEAHEA":
	.quad	0

	.addrsig
