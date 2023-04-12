	.file	"main.cc"
	.text
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	endbr64
	movl	a(%rip), %ecx
	movl	b(%rip), %esi
	cmpl	%esi, %ecx
	jle	.L2
	movl	%ecx, %eax
	addl	%esi, %ecx
	subl	%esi, %eax
	cltd
	idivl	%ecx
	movl	%eax, result(%rip)
.L3:
	xorl	%eax, %eax
	ret
.L2:
	je	.L6
	jge	.L3
	leal	(%rcx,%rcx,2), %eax
	subl	$2, %eax
	cltd
	idivl	%esi
	movl	%eax, result(%rip)
	jmp	.L3
.L6:
	movl	%ecx, %eax
	negl	%eax
	imull	%eax, %ecx
	movl	%ecx, result(%rip)
	jmp	.L3
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.globl	result
	.bss
	.align 4
	.type	result, @object
	.size	result, 4
result:
	.zero	4
	.globl	b
	.align 4
	.type	b, @object
	.size	b, 4
b:
	.zero	4
	.globl	a
	.align 4
	.type	a, @object
	.size	a, 4
a:
	.zero	4
	.ident	"GCC: (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	 1f - 0f
	.long	 4f - 1f
	.long	 5
0:
	.string	 "GNU"
1:
	.align 8
	.long	 0xc0000002
	.long	 3f - 2f
2:
	.long	 0x3
3:
	.align 8
4:
