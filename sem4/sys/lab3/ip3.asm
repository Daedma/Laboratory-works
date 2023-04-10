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
	movl	"?a@@3HA"(%rip), %eax
	movl	"?b@@3HA"(%rip), %r8d
	movl	%eax, %ecx
	subl	%r8d, %ecx
	jle	.LBB0_2
# %bb.1:
	addl	%eax, %r8d
	movl	%ecx, %eax
	jmp	.LBB0_6
.LBB0_2:
	jne	.LBB0_4
# %bb.3:
	imull	%eax, %eax
	negl	%eax
	jmp	.LBB0_7
.LBB0_4:
	jge	.LBB0_8
# %bb.5:
	leal	(%rax,%rax,2), %eax
	addl	$-2, %eax
.LBB0_6:
	cltd
	idivl	%r8d
                                        # kill: def $eax killed $eax def $rax
.LBB0_7:
	movl	%eax, "?result@@3HA"(%rip)
.LBB0_8:
	xorl	%eax, %eax
	retq
                                        # -- End function
	.bss
	.globl	"?a@@3HA"                       # @"?a@@3HA"
	.p2align	2
"?a@@3HA":
	.long	0                               # 0x0

	.globl	"?b@@3HA"                       # @"?b@@3HA"
	.p2align	2
"?b@@3HA":
	.long	0                               # 0x0

	.globl	"?result@@3HA"                  # @"?result@@3HA"
	.p2align	2
"?result@@3HA":
	.long	0                               # 0x0

	.addrsig
