program gauss
  implicit none
  DOUBLE PRECISION, dimension(4,4) ::a, b, c
  DOUBLE PRECISION, dimension(4) :: mnum, l, max,  x
  DOUBLE PRECISION :: lk(4), li(4)
  DOUBLE PRECISION :: ans, maxnum
  INTEGER :: i, j, k(1)
  a = 1

  do i = 1, 4,1
  ! a is matrix, and x is where to start
  ! this is for scaling
    l = ABS(a(:, i))
    maxnum = MAXVAL(l)
    !list for remembering max value
    a(:, i) = a(:, i) / maxnum
    mnum(i) = maxnum
  end do

  do i = 1, 3, 1
    ! this is for pivoting
    max = a(i, :)  ! remember max value
    k = maxloc(max)
    li = a(:, i)
    lk = a(:, k(1))
    a(:,k(1)) = li
    a(:, i) = lk
    
    ! this is for deleteing
    if (a(i,i) /= 0.0 .and. i+1 /= 4) then
      do j = i + 1, 4
        if (a(i,i) /= 0.0 .and. a(i,j) /= 0.0) then
          b(i, j) = -a(i, j) / a(i, i)
          a(:,j) = a(:, j) - (a(i, j)/a(i, i))*a(:,i)
        end if
      end do
    end if
  end do

  ans = 0
  x = 0

!  do i = 4, 1, -1
!    ans = (c(:,i) - SUM(/ (a(j, 1) * x(j) /), j = i, n)/) / a(i,i)
!    x(i) = ans
!    ans = 0
!  end do

  print *, x

end program gauss

