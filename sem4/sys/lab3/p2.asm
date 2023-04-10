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
	movl	"?a@@3HC"(%rip), %eax
	cmpl	"?b@@3HC"(%rip), %eax
	movl	"?a@@3HC"(%rip), %ecx
	movl	"?b@@3HC"(%rip), %edx
	jle	.LBB0_2
# %bb.1:
	subl	%edx, %ecx
	movl	"?a@@3HC"(%rip), %r8d
	addl	"?b@@3HC"(%rip), %r8d
	movl	%ecx, %eax
	cltd
	idivl	%r8d
	jmp	.LBB0_6
.LBB0_2:
	movl	"?a@@3HC"(%rip), %eax
	movl	"?b@@3HC"(%rip), %r8d
	cmpl	%edx, %ecx
	jne	.LBB0_4
# %bb.3:
	imull	%r8d, %eax
	negl	%eax
	jmp	.LBB0_6
.LBB0_4:
	cmpl	%r8d, %eax
	jge	.LBB0_7
# %bb.5:
	movl	"?a@@3HC"(%rip), %eax
	leal	(%rax,%rax,2), %eax
	addl	$-2, %eax
	cltd
	idivl	"?b@@3HC"(%rip)
.LBB0_6:
	movl	%eax, "?result@@3HA"(%rip)
.LBB0_7:
	xorl	%eax, %eax
	retq
                                        # -- End function
	.bss
	.globl	"?a@@3HC"                       # @"?a@@3HC"
	.p2align	2
"?a@@3HC":
	.long	0                               # 0x0

	.globl	"?b@@3HC"                       # @"?b@@3HC"
	.p2align	2
"?b@@3HC":
	.long	0                               # 0x0

	.globl	"?result@@3HA"                  # @"?result@@3HA"
	.p2align	2
"?result@@3HA":
	.long	0                               # 0x0

	.addrsig
	.addrsig_sym "?a@@3HC"
	.addrsig_sym "?b@@3HC"
