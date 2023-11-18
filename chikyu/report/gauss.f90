DOUBLE PRECISION, dimension(4,4) ::a, b, c
DOUBLE PRECISION, dimension(4) ::memo, l, max, x
DOUBLE PRECISION, ::k, ansj
INTEGER, ::i, j
program gauss

do i = 1, 4, 1
! a is matrix, and x is where to start
! this is for scaling
  l = ABS(a(:,i))
  maxnum = maxval(l)
  !list for remembering max value
  a(:, i) = a(:, i) / maxnum
  memo(i) = maxnum
end do

do i = 1, 3, 1
  ! this is for pivoting
  max = a(i, :)  ! remember max value
  k = maxloc(max)
  a(:,(x,k)) = a(:, (k, x))  ! swap
  
  ! this is for deleteing
  if (a(i,i) /= 0.0 .and. x+1 /= n) then
    do j = i + 1, 4
      if (a(i,i) /= 0.0 .and. a(i,j) /= 0.0) then
        b(i, j) = -a(i, j) / a(i, i)
        a(:,j) = a(;, j) - (a(i, j)/a(i, i)*a(:,i)
  endif
end do

ans = 0
x = 0

do i = 4, 1, -1
  ans = (c(;,i) - SUM(/ (a(j, 1) * x(j)), j = i, n)/) / a(i,i)
  x(i) = ans
  ans = 0

print *, x

end program gauss
